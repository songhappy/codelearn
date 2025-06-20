import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from torch.distributed._composable.fsdp import fully_shard
from myutils import (
    get_device_bkd,
    get_xpu_memory_used_from_xpu_smi,
    get_gpu_memory_used_from_nvidia_smi,
    device_type,
)
# ----------------------------
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Setup device helpers
device_type, backend, torch_device = get_device_bkd()

def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if device_type in ["xpu", "cuda"]:
        torch_device.set_device(rank)
        return rank, torch.device(f"{device_type}:{rank}")
    return rank, torch_device


def train():

    rank, device = setup_distributed()

    # Define meta-initialized model
    with torch.device("meta"):
        model = SimpleModel()
    # Apply FSDP sharding to Linear layers
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear):
            fully_shard(module)

    # Fully shard the top module too
    fully_shard(model)
    
    # Dummy dataset
    X = torch.randn(100, 1000)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(X, y)
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=16)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(1):
        sampler.set_epoch(epoch)
        for step, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            # Memory log
            memory_log = (
                get_xpu_memory_used_from_xpu_smi(f"Epoch {epoch}, Step {step}", device.index)
                if device_type == "xpu"
                else get_gpu_memory_used_from_nvidia_smi(f"Epoch {epoch}, Step {step}", device.index)
            )
            print(f"[Rank {rank}] Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}, {memory_log}")

    dist.destroy_process_group()


if __name__ == "__main__":
    train()

