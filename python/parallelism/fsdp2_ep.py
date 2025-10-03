# fsdp2_ep.py
# Launch:
#   torchrun --nproc_per_node 8 fsdp2_ep.py
#
# Layout (DP × EP = 2 × 4 = 8 ranks)
#
#             EP index →     0       1       2       3
# DP index
#       0               [ r0 ]  [ r1 ]  [ r2 ]  [ r3 ]
#       1               [ r4 ]  [ r5 ]  [ r6 ]  [ r7 ]
#
# EP groups (vary EP, fix DP) — 2 groups, size 4 each:
#   EP group @ DP=0: { r0, r1, r2, r3 }
#   EP group @ DP=1: { r4, r5, r6, r7 }
#
# DP groups (vary DP, fix EP) — 4 groups, size 2 each:
#   DP group @ EP=0: { r0, r4 }
#   DP group @ EP=1: { r1, r5 }
#   DP group @ EP=2: { r2, r6 }
#   DP group @ EP=3: { r3, r7 }
#
# Mental model for one DP row (same microbatch on all 4 EP ranks):
#    r0(ep0)   r1(ep1)   r2(ep2)   r3(ep3)
#       │         │         │         │
#      fc1      fc1       fc1       fc1   (replicated across EP, sharded across DP by FSDP)
#       │         │         │         │
#     router   router    router    router  (replicated across EP, same top-1 routing)
# ┌─────┴────────┴────────┴────────┴─────┐
# │     local      local      local      │
# │    experts    experts    experts     │  (each EP rank owns a slice of experts)
# └────────────── all_reduce(SUM) ─────────
#                    (within EP group)
#                  each rank gets full MoE output
#       │         │         │         │
#      fc2      fc2       fc2       fc2   (replicated across EP, sharded across DP by FSDP)
#       │         │         │         │
#     loss      loss      loss      loss
#
# Notes:
# - FSDP shards parameters across the DP dimension only (dp_mesh).
# - EP is handled manually inside the MoE layer with an all_reduce over the EP group.
# - Routing is top-1, non-differentiable (demo purpose only).
# - Real MoE often uses all_to_all token exchange; we keep it simple here.

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh

# FSDP2 (composable API)
from torch.distributed._composable.fsdp import fully_shard

def get_backend_and_device():
    devtype = "cuda" if torch.cuda.is_available() else ("xpu" if hasattr(torch, "xpu") else "cpu")
    backend = "nccl" if devtype == "cuda" else ("xccl" if devtype == "xpu" else "gloo")
    return backend, devtype

# -----------------------------
# Simple Expert-Parallel MoE Layer (EP + all_reduce gather)
# -----------------------------
class MoELayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_experts, ep_mesh: DeviceMesh):
        super().__init__()
        self.out_dim = out_dim
        self.ep_mesh = ep_mesh
        # EP group is "same DP index, vary EP index"
        self.ep_group = ep_mesh.get_group()
        self.ep_rank = dist.get_rank(self.ep_group)
        ep_world = ep_mesh.size()
        assert num_experts % ep_world == 0, "num_experts must be divisible by EP world size"
        self.experts_per_rank = num_experts // ep_world
        self.offset = self.ep_rank * self.experts_per_rank

        # Simple top-1 router (frozen / non-differentiable for demo)
        self.router = nn.Linear(in_dim, num_experts, bias=False)
        for p in self.router.parameters():
            p.requires_grad_(False)

        # Experts owned by THIS EP rank only
        self.local_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
            for _ in range(self.experts_per_rank)
        ])

    def forward(self, x):
        # Decide which expert handles each token (no grad for routing)
        with torch.no_grad():
            expert_ids = self.router(x).argmax(dim=-1)  # [B]
            local_mask = (expert_ids // self.experts_per_rank) == self.ep_rank
            local_ids = expert_ids[local_mask] - self.offset  # remap to [0..experts_per_rank-1] on this rank

        # Compute outputs for tokens owned by my local experts
        y = x.new_zeros(x.size(0), self.out_dim)
        if local_mask.any():
            xin = x[local_mask]
            for i in range(self.experts_per_rank):
                sel = (local_ids == i)
                if sel.any():
                    y_local = self.local_experts[i](xin[sel])
                    # Write into the corresponding rows
                    y[local_mask][sel] = y_local

        # Merge across EP group. Only one rank wrote each row; SUM is fine.
        dist.all_reduce(y, group=self.ep_group, op=dist.ReduceOp.SUM)
        return y

# -----------------------------
# Model = dense → MoE → dense
# -----------------------------
class ToyMoE(nn.Module):
    def __init__(self, ep_mesh):
        super().__init__()
        self.fc1 = nn.Linear(32, 16)
        self.moe = MoELayer(16, 32, 16, num_experts=4, ep_mesh=ep_mesh)  # 4 experts total, 1 per EP rank
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.moe(x)
        return self.fc2(x)

# -----------------------------
# Training loop (DP × EP over a 2×4 mesh)
# -----------------------------
def train(rank, world_size, local_rank):
    backend, devtype = get_backend_and_device()
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    # Bind this rank to its device
    if devtype == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif devtype == "xpu":
        try:
            torch.xpu.set_device(local_rank)
        except Exception:
            pass
        device = torch.device("xpu", local_rank)
    else:
        device = torch.device("cpu")

    # 2D mesh: DP × EP = 2 × 4 = 8 ranks
    assert world_size == 8, "This example expects 8 ranks (2×4)."
    mesh = init_device_mesh(devtype, (2, 4), mesh_dim_names=["dp", "ep"])
    dp_mesh, ep_mesh = mesh["dp"], mesh["ep"]

    # Info for DP data sharding
    dp_group = dp_mesh.get_group()
    dp_rank  = dist.get_rank(dp_group)
    dp_size  = dp_mesh.size()

    # Build model on device
    model = ToyMoE(ep_mesh).to(device)

    # Wrap the whole model with FSDP across the DP mesh only
    fully_shard(model, mesh=dp_mesh)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Synthetic dataset on this device
    x = torch.randn(256, 32, device=device)
    y = torch.randint(0, 10, (256,), device=device)
    ds = TensorDataset(x, y)

    # Only DP ranks participate in data partitioning (EP ranks share the same batch within a DP row)
    sampler = torch.utils.data.distributed.DistributedSampler(
        ds, num_replicas=dp_size, rank=dp_rank, shuffle=True, drop_last=False
    )
    loader = DataLoader(ds, batch_size=32, sampler=sampler, pin_memory=False)

    for epoch in range(3):
        sampler.set_epoch(epoch)
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss)
        if dp_rank == 0:
            # (Print once per DP row.)
            print(f"[Epoch {epoch+1}] Loss: {running:.4f}")

    dist.destroy_process_group()

def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    train(rank, world_size, local_rank)

if __name__ == "__main__":
    main()
