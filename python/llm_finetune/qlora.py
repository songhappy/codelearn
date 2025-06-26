import torch
import torch.nn as nn
import torch.optim as optim
from torchtune.modules.peft.lora import LoRALinear
import os
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.distributed._composable.fsdp import fully_shard


# Simple dummy dataset
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=512):
        self.x = torch.randn(size, 16)
        self.y = torch.randint(0, 2, (size,))

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

# Simple model using two quantized LoRALinear layers
class SimpleLoRAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = LoRALinear(16, 32, rank=4, alpha=8, quantize_base=True, scaler_block_size=1)
        self.relu = nn.ReLU()
        self.l2 = LoRALinear(32, 2, rank=4, alpha=8, quantize_base=True, scaler_block_size=1)


    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))


def print_param_shapes(model):
    print("Trainable Parameters:")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {n}: {tuple(p.shape)}")
    print("Non-trainable Parameters:")
    for n, p in model.named_parameters():
        if not p.requires_grad:
            print(f"  {n}: {tuple(p.shape)}")


def train(local_rank, world_size, distributed=False):
    device_type = "cuda" if torch.cuda.is_available() else "xpu"
    device = torch.device(f"{device_type}:{local_rank}")

    if device_type == "cuda":
        torch.cuda.set_device(device)
    else:
        torch.xpu.set_device(device)

    model = SimpleLoRAModel().to(device)

    if distributed:
        dist.barrier()
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                fully_shard(module)
        print(f"[Rank {local_rank}] FSDP2 wrapping done.")

    print_param_shapes(model)

    dataset = DummyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank) if distributed else None
    dataloader = DataLoader(dataset, batch_size=32, shuffle=(sampler is None), sampler=sampler)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(3):
        if distributed:
            sampler.set_epoch(epoch)
        total_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Rank {local_rank}] Epoch {epoch+1}, Loss: {total_loss:.4f}")


def main():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        backend = "xccl" if hasattr(torch, "xpu") and torch.xpu.is_available() else "nccl"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        train(local_rank, world_size, distributed=True)
        dist.destroy_process_group()
    else:
        print("Running in single-device mode")
        train(local_rank=0, world_size=1, distributed=False)


if __name__ == "__main__":
    main()