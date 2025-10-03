# torchrun --nproc_per_node 8 fsdp_pp.py
# nohup nvidia-smi -l 1 --query-gpu=memory.used,memory.free,memory.total --format=csv > /home/songhappy/git/codelearn/python/ddp/memo.txt 2>&1 &

# Topology: DP = 4, PP = 2  (world_size = DP * PP = 8)
#
# | GPU Rank | DP Rank | PP Stage | Input Portion Received    | Weights Held (FSDP shards across DP replicas of that stage) |
# | -------- | ------- | -------- | ------------------------- | ------------------------------------------------------------ |
# | **0**    | 0       | 0        | ¼ of dataset (DP row 0)   | Stage 0 params: 1/4 shard                                    |
# | **1**    | 0       | 1        | same as rank 0 (DP row 0) | Stage 1 params: 1/4 shard                                    |
# | **2**    | 1       | 0        | ¼ of dataset (DP row 1)   | Stage 0 params: 1/4 shard                                    |
# | **3**    | 1       | 1        | same as rank 2 (DP row 1) | Stage 1 params: 1/4 shard                                    |
# | **4**    | 2       | 0        | ¼ of dataset (DP row 2)   | Stage 0 params: 1/4 shard                                    |
# | **5**    | 2       | 1        | same as rank 4 (DP row 2) | Stage 1 params: 1/4 shard                                    |
# | **6**    | 3       | 0        | ¼ of dataset (DP row 3)   | Stage 0 params: 1/4 shard                                    |
# | **7**    | 3       | 1        | same as rank 6 (DP row 3) | Stage 1 params: 1/4 shard                                    |
#
# Notes:
# - Pipeline parallelism has 2 stages (0 -> 1). Microbatches flow left→right via point-to-point transfers.
# - FSDP2 sharding happens ACROSS DP replicas but WITHIN the SAME STAGE (i.e., per column above), 4-way in this example.
# - Data is partitioned by DP rows (¼ each). Only Stage 0 ranks load inputs/targets; Stage 1 ranks just participate in the pipeline.
# - Each rank holds only a shard of ITS STAGE’S parameters (due to FSDP2); it does NOT hold weights from other stages.

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import pipeline
from torch.distributed.pipelining.stage import StageModule


# -------------------------
# Model split into 2 stages
# -------------------------

class Stage0(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1024, 128)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear1(x))


class Stage1(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear2 = nn.Linear(128, 256)

    def forward(self, x):
        return self.linear2(x)  # logits (N, 256)


# -------------------------
# Data loader for DP rows
# -------------------------

def create_dataloader(dp_rank: int, dp_world_size: int, batch_size=32, device="cuda"):
    # Synthetic classification data
    N = 1000
    x = torch.randn(N, 1024, device=device)
    y = torch.randint(0, 256, (N,), device=device)

    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset, num_replicas=dp_world_size, rank=dp_rank, drop_last=False)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    # Compute expected number of steps so non-input stages can iterate pipe(None) the same number of times
    per_replica = math.ceil(N / dp_world_size)
    steps = math.ceil(per_replica / batch_size)
    return loader, steps


# -------------------------
# Rank mapping helpers
# -------------------------

def get_coords(rank: int, pp: int):
    """Return (dp_idx, pp_idx) given global rank and PP size."""
    dp_idx = rank // pp
    pp_idx = rank % pp
    return dp_idx, pp_idx


def make_pipeline_group(dp_idx: int, pp: int):
    """Ranks in the same DP row form a PP chain: [dp_idx*pp + 0, dp_idx*pp + 1, ...]."""
    ranks = [dp_idx * pp + s for s in range(pp)]
    return dist.new_group(ranks), ranks


def make_fsdp_mesh_for_stage(pp_idx: int, dp: int, pp: int, device_type: str):
    """For a fixed stage (pp_idx), FSDP2 shards that stage across all DP replicas (the column)."""
    ranks = [d * pp + pp_idx for d in range(dp)]  # column for this stage
    mesh = DeviceMesh(device_type, ranks)         # 1D mesh for FSDP2 on this stage
    return mesh, ranks


# -------------------------
# Train
# -------------------------

def train(rank: int, world_size: int, local_rank: int):
    backend = "nccl" if torch.cuda.is_available() else "xccl"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    device_type = "cuda" if torch.cuda.is_available() else "xpu"
    device = torch.device(device_type, local_rank)
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)

    torch.manual_seed(0)

    # Set PP and derive DP from world_size
    PP = 2  # number of pipeline stages
    assert world_size % PP == 0, "world_size must be divisible by PP"
    DP = world_size // PP

    dp_idx, pp_idx = get_coords(rank, PP)

    # Create process groups:
    # - pipeline_group: the 2 ranks in this DP replica that form the pipeline chain
    # - fsdp_mesh (DeviceMesh): the 4 ranks holding the SAME STAGE across all DP replicas
    pipeline_group, pipeline_group_ranks = make_pipeline_group(dp_idx, PP)
    fsdp_mesh, fsdp_group_ranks = make_fsdp_mesh_for_stage(pp_idx, DP, PP, device_type)

    if dist.get_rank() == 0:
        print(f"World Size={world_size}  DP={DP}  PP={PP}")
        print(f"Example groups -> pipeline_group for DP row 0: {[0,1]}, fsdp groups per stage: "
              f"Stage0: {[0,2,4,6]}, Stage1: {[1,3,5,7]}")

    # Build the local pipeline stage for this rank
    if pp_idx == 0:
        stage = Stage0().to(device)
    else:
        stage = Stage1().to(device)

    # Wrap the stage in FSDP2; sharding happens ACROSS DP replicas of this stage (fsdp_mesh)
    stage = fully_shard(stage, mesh=fsdp_mesh)

    # Wrap stage for pipeline and assemble pipeline on the per-DP pipeline group
    stage_mod = StageModule(stage, stage_index=pp_idx, num_stages=PP).to(device)
    pipe = pipeline(
        stage_mod,
        num_stages=PP,
        group=pipeline_group,
        chunks=4,  # microbatches
    )

    # Optimizer for local stage params
    optimizer = optim.Adam(stage.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Only Stage 0 needs a DataLoader; others just iterate the same number of steps and call pipe(None)
    if pp_idx == 0:
        dataloader, steps = create_dataloader(dp_idx, DP, batch_size=32, device=device)
    else:
        # Make sure non-input stages know "steps"
        # Use the same formula so it's deterministic everywhere
        _, steps = create_dataloader(dp_idx, DP, batch_size=32, device=device)

    EPOCHS = 5
    for epoch in range(EPOCHS):
        total_loss = 0.0

        if pp_idx == 0:
            for x, y in dataloader:
                optimizer.zero_grad(set_to_none=True)
                # Forward through the pipeline, returns final logits to Stage 0
                logits = pipe(x)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        else:
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                # Participate in pipeline without owning inputs/targets
                pipe(None)
                optimizer.step()

        # Simple logging from one rank per DP row (stage 0)
        if pp_idx == 0 and dp_idx == 0:
            print(f"[Epoch {epoch+1}] Loss (DP row 0): {total_loss:.4f}")

        dist.barrier()

    dist.destroy_process_group()


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    train(rank, world_size, local_rank)


if __name__ == "__main__":
    main()
