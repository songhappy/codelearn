import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os

# ----- Toy Model -----
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1024, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 256)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# ----- Dataset -----
def create_dataloader(rank, world_size, batch_size=32):
    x = torch.randn(1000, 1024)
    y = torch.randint(0, 256, (1000,))
    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

# ----- Training Loop -----
def train(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = ToyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    dataloader = create_dataloader(rank, world_size)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        total_loss = 0.0

        dataloader.sampler.set_epoch(epoch)  # Ensure shuffling is different every epoch

        for x, y in dataloader:
            x, y = x.to(rank), y.to(rank)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if rank == 0:
            print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    dist.destroy_process_group()

# ----- Entry Point -----
def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    train(rank, world_size, local_rank)

if __name__ == "__main__":
    main()
