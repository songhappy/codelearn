# export WORLD_SIZE=8
# export TP_SIZE=2
# export EP_SIZE=2   # DP will be inferred as 2
# torchrun --nproc_per_node=$WORLD_SIZE 3d_fsdp_tp_ep_simple.py

# 1) Dataset / Input
# Sharded across DP, replicated across TP×EP within each DP row.
# DP=0 layer (microbatch A)    DP=1 layer (microbatch B)
# [t0 e0][t0 e1]               [t0 e0][t0 e1]
# [t1 e0][t1 e1]               [t1 e0][t1 e1]
# ^ TP rows  ^ EP cols         ^ TP rows  ^ EP cols

# 2) linear1 (Colwise TP; FSDP on DP; replicated across EP)
# Split across TP (col-wise shards per t0/t1)
# Sharded across DP (FSDP)
# Replicated across EP
# DP=0:  [ t0| t1 ]  [ t0| t1 ]   (EP=0, EP=1)
# DP=1:  [ t0| t1 ]  [ t0| t1 ]

# 3) Router (inside MoE)
# Two possibilities depending on how you configure it:

# A) As in your simplified mental model (router left unsharded by TP/DP):
# Replicated across TP×EP within a DP row (identical routing).
# (Often you still FSDP this; see B.)
# DP=0:  [ t0 ][ t0 ]     (each is same router W)
#        [ t1 ][ t1 ]
# DP=1:  [ t0 ][ t0 ]
#        [ t1 ][ t1 ]

# B) If you keep fully_shard on the router Linear (common in the code):
# Sharded across DP, replicated across TP×EP.
# Net effect on routing stays identical within each DP row because input is the same.

# 4) Experts (MoE local experts)
# Split across EP (each EP column owns a distinct subset of experts)
# Sharded across DP (FSDP on each expert’s Linear layers)
# Not TP-sharded → effectively replicated across TP (since MoE isn’t TP’d)
# EP=0 column owns its experts            EP=1 column owns its experts
# DP=0: [ t0 ][ t1 ]   (same params on t0/t1; DP=0 shard)
# DP=1: [ t0 ][ t1 ]   (same params on t0/t1; DP=1 shard)
# So per expert: EP-split, DP-sharded, TP-replicated.

# 5) linear2 (Rowwise TP; FSDP on DP; replicated across EP)
# Split across TP (row-wise shards per t0/t1)
# Sharded across DP (FSDP)
# Replicated across EP

# 6) MoE EP merge (communications picture)
# Inside a (DP,TP) fiber (fix DP & TP; vary EP):
# (local expert outputs + zeros)  +  (zeros + local expert outputs)
#                     └─ EP all_reduce(SUM) ─┘
# → every EP rank in the fiber has the full MoE output
# Group: EP group at fixed (DP,TP)
# This works because top-1 routing writes each token’s row on exactly one EP rank.

# 7) Loss
# Replicated across TP×EP within a DP row (same logits after EP reduce).
# Computed redundantly on each rank in the fiber.
# export WORLD_SIZE=8
# export TP_SIZE=2
# export EP_SIZE=2   # DP will be inferred as 2
# torchrun --nproc_per_node=$WORLD_SIZE 3d_fsdp_tp_ep_simple.py

import os, sys, time, datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset

from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed.tensor.placement_types import Replicate
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh

# --------------- DEBUG ENV ---------------
def set_debug_env():
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    # oneCCL (XPU)
    os.environ.setdefault("XCCL_LOG_LEVEL", "info")
set_debug_env()

def get_backend_and_device():
    device_type = "cuda" if torch.cuda.is_available() else ("xpu" if hasattr(torch, "xpu") else "cpu")
    if device_type == "cuda":
        backend = "nccl"
    elif device_type == "xpu":
        backend = "xccl"
    else:
        backend = "gloo"
    return backend, device_type

def infer_mesh_shape(world_size: int):
    tp = int(os.environ.get("TP_SIZE", "2"))
    ep = int(os.environ.get("EP_SIZE", "2"))
    dp_env = os.environ.get("DP_SIZE")
    if dp_env is not None:
        dp = int(dp_env)
    else:
        assert (tp * ep) > 0 and world_size % (tp * ep) == 0, (
            f"WORLD_SIZE={world_size} not divisible by TP*EP={tp*ep}. "
            f"Set DP_SIZE/TP_SIZE/EP_SIZE explicitly."
        )
        dp = world_size // (tp * ep)
    if dp * tp * ep != world_size:
        raise ValueError(
            f"Invalid mesh: DP({dp})*TP({tp})*EP({ep}) != WORLD_SIZE({world_size}). "
            f"Set DP_SIZE/TP_SIZE/EP_SIZE so the product matches WORLD_SIZE."
        )
    return dp, tp, ep

def get_mesh_ranks(mesh: DeviceMesh):
    group = mesh.get_group()
    return dist.get_rank(group), mesh.size()

# --------------- EP GROUP (explicit) ---------------
def build_ep_group(mesh3d: DeviceMesh):
    """
    Build an EP group explicitly from the mesh's rank layout to avoid API/version ambiguity.
    Group = all ranks with same (dp,tp), varying ep.
    """
    # mesh3d._mesh is a tensor of global ranks shaped (dp,tp,ep) in recent PyTorch;
    # if not present, fallback to mesh3d.mesh
    rank_layout = getattr(mesh3d, "_mesh", getattr(mesh3d, "mesh"))
    # Find my (dp,tp,ep) coordinate
    my_global = dist.get_rank()
    coords = (rank_layout == my_global).nonzero(as_tuple=False).squeeze().tolist()
    if not isinstance(coords, list):
        coords = coords.tolist()
    dp_i, tp_i, ep_i = coords
    ep_ranks = rank_layout[dp_i, tp_i, :].tolist()
    pg = dist.new_group(ep_ranks)
    return pg, (dp_i, tp_i, ep_i), ep_ranks

# -----------------------------
# Simplified Expert-Parallel MoE layer (Top-1 + EP all_reduce)
# -----------------------------
class MoELayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_experts, ep_group):
        super().__init__()
        self.out_dim = out_dim
        self.ep_group = ep_group
        self.ep_world = dist.get_world_size(self.ep_group)
        self.ep_rank = dist.get_rank(self.ep_group)
        assert num_experts % self.ep_world == 0, "num_experts must be divisible by EP world size"
        self.experts_per_rank = num_experts // self.ep_world
        self.offset = self.ep_rank * self.experts_per_rank

        self.router = nn.Linear(in_dim, num_experts, bias=False)
        for p in self.router.parameters():
            p.requires_grad_(False)

        self.local_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
            for _ in range(self.experts_per_rank)
        ])

    def forward(self, x):
        # [DBG] minimal heartbeat
        # print(f"EP{self.ep_rank}: MoE fwd enter", flush=True)

        with torch.no_grad():
            expert_ids = self.router(x).argmax(dim=-1)  # [B]
            local_mask = (expert_ids // self.experts_per_rank) == self.ep_rank
            local_ids = expert_ids[local_mask] - self.offset

        y = torch.zeros(x.size(0), self.out_dim, device=x.device, dtype=x.dtype)

        if local_mask.any():
            rows_all = local_mask.nonzero(as_tuple=True)[0]
            for i in range(self.experts_per_rank):
                sel = (local_ids == i)
                if sel.any():
                    rows = rows_all[sel]
                    xin = x.index_select(0, rows)
                    y_local = self.local_experts[i](xin)
                    y.index_copy_(0, rows, y_local)

        # Barrier before collective to identify stalls
        dist.barrier(self.ep_group)
        dist.all_reduce(y, group=self.ep_group, op=dist.ReduceOp.SUM)
        return y

class ToyMoE(nn.Module):
    def __init__(self, ep_group, num_experts=4, d_in=32, d_hidden=16, d_moe_hidden=32, d_out=10):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_hidden)
        self.moe = MoELayer(d_hidden, d_moe_hidden, d_hidden, num_experts=num_experts, ep_group=ep_group)
        self.linear2 = nn.Linear(d_hidden, d_out)
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.moe(x)
        return self.linear2(x)

def create_dataloader(dp_rank: int, dp_size: int, batch_size: int = 32, device: torch.device = torch.device("cuda")):
    x = torch.randn(256, 32, device=device)
    y = torch.randint(0, 10, (256,), device=device)
    dataset = TensorDataset(x, y)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=dp_size, rank=dp_rank, shuffle=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=False)

def barrier_mark(tag, group=None):
    # Numbered barriers to pinpoint hangs
    dist.barrier(group=group)
    if dist.get_rank() == 0:
        print(f"[BARRIER {tag}] passed", flush=True)

def train(rank: int, world_size: int, local_rank: int):
    backend, device_type = get_backend_and_device()
    timeout = datetime.timedelta(seconds=int(os.environ.get("DIST_TIMEOUT", "600")))
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, timeout=timeout)
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif device_type == "xpu":
        try:
            import torch.xpu as xpu
            xpu.set_device(local_rank)
        except Exception as e:
            print(f"[WARN] xpu.set_device failed: {e}", flush=True)
        device = torch.device("xpu", local_rank)
    else:
        device = torch.device("cpu")

    torch.manual_seed(0)
    if rank == 0:
        print(f"[R{rank}] backend={backend} device_type={device_type}", flush=True)

    # Build 3D mesh
    dp, tp, ep = infer_mesh_shape(world_size)
    mesh_3d = init_device_mesh(device_type, (dp, tp, ep), mesh_dim_names=["dp", "tp", "ep"])
    barrier_mark("MESH-INIT")
    dp_mesh = mesh_3d["dp"]
    tp_mesh = mesh_3d["tp"]
    ep_mesh = mesh_3d["ep"]

    ep_group = ep_mesh.get_group()

    dp_rank, dp_size = get_mesh_ranks(dp_mesh)
    if rank == 0:
        print(f"3D mesh: DP={dp_size}, TP={tp_mesh.size()}, EP={ep_mesh.size()}", flush=True)

    # Explicit EP group (avoids version quirks)
    ep_group, coords, ep_ranks = build_ep_group(mesh_3d)
    if rank == 0:
        print(f"[INFO] EP group example ranks={ep_ranks}", flush=True)
    barrier_mark("EP-GROUP", ep_group)

    # Build model
    num_experts = int(os.environ.get("NUM_EXPERTS", "4"))
    assert num_experts % ep_mesh.size() == 0
    model = ToyMoE(ep_group=ep_group, num_experts=num_experts).to(device)
    barrier_mark("MODEL-BUILT")

    # TP on dense layers only
    tp_plan = {
        "linear1": ColwiseParallel(output_layouts=Replicate()),
        "linear2": RowwiseParallel(input_layouts=Replicate()),
    }
    model = parallelize_module(model, tp_mesh, tp_plan)
    barrier_mark("TP-DONE")

    # FSDP across DP for Linear params
    def shard_condition(name: str, module: nn.Module):
        return isinstance(module, nn.Linear)
    for name, module in reversed(list(model.named_modules())):
        if shard_condition(name, module):
            fully_shard(module, mesh=dp_mesh)
    fully_shard(model, mesh=dp_mesh)
    barrier_mark("FSDP-DONE")

    dataloader = create_dataloader(dp_rank, dp_mesh.size(), device=device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    num_epochs = int(os.environ.get("EPOCHS", "1"))
    for epoch in range(num_epochs):
        if isinstance(dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        for x, y in dataloader:
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        if dp_rank == 0:
            print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}", flush=True)

    barrier_mark("END")
    dist.destroy_process_group()

def main():
    rank = int(os.environ["RANK"]); world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    train(rank, world_size, local_rank)

if __name__ == "__main__":
    main()
