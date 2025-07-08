import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
import torch.nn as nn
import traceback

class SimpleModel(nn.Module):
    def __init__(self, input_size=100, hidden_size=50, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.net(x)

def setup():
    dist.init_process_group("nccl")
    
def cleanup():
    dist.destroy_process_group()

def train():
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"Running on rank {rank} with local_rank {local_rank} and world_size {world_size}")
    except KeyError as e:
        raise RuntimeError(f"Missing environment variable: {e}. Are you running with torchrun?")

    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"Invalid local_rank={local_rank}. Available CUDA devices: {torch.cuda.device_count()}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    setup()

    # Dummy data
    x = torch.randn(1000, 100)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    model = SimpleModel().to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(3):
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        if rank == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.4f}")

    cleanup()

if __name__ == "__main__":
    try:
        train()
    except Exception:
        print("Exception occurred in simple_example.py")
        traceback.print_exc()
        exit(1)
