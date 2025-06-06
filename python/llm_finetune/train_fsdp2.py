#torchrun --nproc_per_node=2 train_fsdp2.py
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import gc
from torch.distributed._composable.fsdp import fully_shard
# from torch.distributed.fsdp import fully_shard 

# Dummy dataset
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000):
        self.x = torch.randn(size, 10)
        self.y = torch.randint(0, 2, (size,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Simple model
class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.seq(x)

def train(local_rank, world_size, distributed=False):
    device = torch.device(f"xpu:{local_rank}")
    torch.xpu.set_device(device)

    model = SmallModel().to(device)

    if distributed:
        dist.barrier()
        rank = dist.get_rank()
        print(rank)
        # Define a simple example condition: shard all Linear layers
        def shard_condition(name, module):
            return isinstance(module, nn.Linear)

        num_layers_sharded = 0
        for name, module in reversed(list(model.named_modules())):
            if shard_condition(name, module):
                fully_shard(module)
                num_layers_sharded += 1
        print(f"[Rank {local_rank}] Manually sharded {num_layers_sharded} submodules.")

    dataset = DummyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True) if distributed else None
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32, shuffle=(sampler is None))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(3):
        if distributed:
            sampler.set_epoch(epoch)
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            print("before loss.backward")
            gc.collect()
            torch.xpu.empty_cache()
            loss.backward()
            print("after loss.backward")
            gc.collect()
            torch.xpu.empty_cache()
            optimizer.step()
            print("after optimizer.step")
            gc.collect()
            torch.xpu.empty_cache()
        if not distributed or local_rank == 0:
            print(f"[Rank {local_rank}] Epoch {epoch+1}, Loss: {loss.item():.4f}")

def main():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Distributed mode
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="xccl", rank=rank, world_size=world_size)
        train(local_rank, world_size, distributed=True)
        dist.destroy_process_group()
    else:
        # Single-card mode
        print("Running in single-XPU mode")
        train(local_rank=0, world_size=1, distributed=False)

if __name__ == "__main__":
    main()