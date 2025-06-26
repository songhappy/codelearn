import os
import gc
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed._composable.fsdp import fully_shard

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

def get_device_and_backend():
    if torch.backends.mps.is_available():
        return "mps", None  # Apple M1/M2
    elif torch.cuda.is_available():
        return "cuda", "nccl"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu", "xccl"
    else:
        raise RuntimeError("No supported device found (CUDA or XPU required).")

def train(local_rank, world_size, distributed=False, device_type="cuda"):
    device = torch.device(f"{device_type}:{local_rank}")

    # Set device for current process
    if device_type == "xpu":
        torch.xpu.set_device(device)
    elif device_type == "cuda":
        torch.cuda.set_device(device)

    model = SmallModel().to(device)

    if distributed:
        dist.barrier()
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

            # Clear cache depending on device
            if device_type == "xpu":
                torch.xpu.empty_cache()
            elif device_type == "cuda":
                torch.cuda.empty_cache()

            loss.backward()
            print("after loss.backward")
            if device_type == "xpu":
                torch.xpu.empty_cache()
            elif device_type == "cuda":
                torch.cuda.empty_cache()

            optimizer.step()
            print("after optimizer.step")
            if device_type == "xpu":
                torch.xpu.empty_cache()
            elif device_type == "cuda":
                torch.cuda.empty_cache()

        if not distributed or local_rank == 0:
            print(f"[Rank {local_rank}] Epoch {epoch+1}, Loss: {loss.item():.4f}")

def main():
    device_type, backend = get_device_and_backend()

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Distributed mode
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        train(local_rank, world_size, distributed=True, device_type=device_type)
        dist.destroy_process_group()
    else:
        print(f"Running in single-{device_type.upper()} mode")
        train(local_rank=0, world_size=1, distributed=False, device_type=device_type)

if __name__ == "__main__":
    main()
