# torchrun --nproc_per_node 8 fsdp_tp.py 
 # nohup nvidia-smi -l 1 --query-gpu=memory.used,memory.free,memory.total --format=csv > /home/songhappy/git/codelearn/python/ddp/memo.txt 2>&1 &

# | GPU Rank | DP Rank | TP Rank | Input Portion Received    | Weights Held                            |
# | -------- | ------- | ------- | ------------------------- | --------------------------------------- |
# | **0**    | 0       | 0       | ¼ of dataset (DP row 0)   | `linear1`: columns 0, `linear2`: rows 0 |
# | **1**    | 0       | 1       | same as rank 0 (DP row 0) | `linear1`: columns 1, `linear2`: rows 1 |
# | **2**    | 1       | 0       | ¼ of dataset (DP row 1)   | `linear1`: columns 0, `linear2`: rows 0 |
# | **3**    | 1       | 1       | same as rank 2 (DP row 1) | `linear1`: columns 1, `linear2`: rows 1 |
# | **4**    | 2       | 0       | ¼ of dataset (DP row 2)   | `linear1`: columns 0, `linear2`: rows 0 |
# | **5**    | 2       | 1       | same as rank 4 (DP row 2) | `linear1`: columns 1, `linear2`: rows 1 |
# | **6**    | 3       | 0       | ¼ of dataset (DP row 3)   | `linear1`: columns 0, `linear2`: rows 0 |
# | **7**    | 3       | 1       | same as rank 6 (DP row 3) | `linear1`: columns 1, `linear2`: rows 1 |


# Each of the 8 ranks holds only 1/8 of the model params (TP sharding × FSDP sharding).
# During training:
# TP collectives happen within columns (2-way).
# FSDP collectives happen within rows (4-way).
# Data is split across rows (DP), not across columns (TP).

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset

from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1024, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 256)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


def create_dataloader(rank, world_size, batch_size=32, device="cuda"):
    x = torch.randn(1000, 1024, device=device)
    y = torch.randint(0, 256, (1000,), device=device)
    dataset = TensorDataset(x, y)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def train(rank, world_size, local_rank):
    backend = "nccl" if torch.cuda.is_available() else "xccl"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    device_type = "cuda" if torch.cuda.is_available() else "xpu"
    device = torch.device(device_type, local_rank)
    torch.cuda.set_device(local_rank) if device_type == "cuda" else None

    # 2D mesh: 2 DP × 2 TP = 4 total ranks
    mesh_2d = init_device_mesh(device_type, (4, 2), mesh_dim_names=["dp", "tp"])

    tp_mesh = mesh_2d["tp"]
    dp_mesh = mesh_2d["dp"]
    model = ToyModel()
    tp_plan = {
        'linear1': ColwiseParallel(),
        'linear2': RowwiseParallel()
    }
    model = parallelize_module(model, tp_mesh, tp_plan)

    # Apply FSDP sharding on top of TP using full mesh
    def shard_condition(name, module):
        return isinstance(module, nn.Linear)


    num_layers_sharded = 0
    for name, module in reversed(list(model.named_modules())):
        if shard_condition(name, module):
            fully_shard(module, mesh=dp_mesh)
            num_layers_sharded += 1
    fully_shard(model, mesh=dp_mesh)

    if dist.get_rank() == 0:
        print(f"[Rank {local_rank}] Manually sharded {num_layers_sharded} submodules.")

    dataloader = create_dataloader(rank, world_size, device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            out = model(x)
            loss = loss_fn(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if dist.get_rank() == 0:
            print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    dist.destroy_process_group()


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    train(rank, world_size, local_rank)


if __name__ == "__main__":
    main()
