# torchrun --nproc_per_node 4 ddp.py
# each rank independently creates its own random 1000-sample datase, 4 ranks gives 4000 samples
# Effective global batch size = per_rank_batch_size × world_size = 32 × 4 = 128
# torchrun --nproc_per_node 4 ddp.py

import os, torch, torch.nn as nn, torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1024, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 256)
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

def create_shared_dataloader(rank, world_size, batch_size=32):
    # Everyone creates the SAME tensors via fixed seed (simple demo)
    torch.manual_seed(1234)
    x = torch.randn(1000, 1024)
    y = torch.randint(0, 256, (1000,))
    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

def train(rank, world_size, local_rank):
    dist.init_process_group("xccl", rank=rank, world_size=world_size)
    torch.xpu.set_device(local_rank)

    model = ToyModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    dataloader = create_shared_dataloader(rank, world_size, batch_size=32)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        total_loss = 0.0
        dataloader.sampler.set_epoch(epoch)
        for x, y in dataloader:
            x, y = x.to(local_rank), y.to(local_rank)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()        # DDP averages grads across ranks
            opt.step()
            total_loss += loss.item()
        if rank == 0:
            print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")
    dist.destroy_process_group()

def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    train(rank, world_size, local_rank)

if __name__ == "__main__":
    main()
