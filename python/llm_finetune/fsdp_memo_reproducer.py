import os
import json
import torch
import subprocess
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from torch.distributed._composable.fsdp import fully_shard

# ----------------------------
# Device + Backend Detection
# ----------------------------
if torch.xpu.is_available():
    torch_device = torch.xpu
    backend = "xpu:xccl"
    device_type = "xpu"
elif torch.cuda.is_available():
    torch_device = torch.cuda
    backend = "nccl"
    device_type = "cuda"
else:
    torch_device = torch.device("cpu")
    backend = "gloo"
    device_type = "cpu"

# ----------------------------
# Setup Distributed
# ----------------------------
def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if device_type in ["xpu", "cuda"]:
        torch_device.set_device(rank)
        return rank, torch.device(f"{device_type}:{rank}")
    return rank, torch_device

# ----------------------------
# Memory Monitor Helpers
# ----------------------------
def get_xpu_memory_used_from_xpu_smi(tag, device_id=0):
    if device_type != "xpu":
        return
    try:
        result = subprocess.run(
            ["xpu-smi", "stats", "-d", str(device_id), "-j"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        stats = json.loads(result.stdout)
        tile_level = stats.get("tile_level", [])
        total_mem_mb = sum(metric["value"] for tile in tile_level for metric in tile.get("data_list", []) if metric["metrics_type"] == "XPUM_STATS_MEMORY_USED")
        return f"[{tag}] xpu-smi memory used (device {device_id}): {total_mem_mb / 1024:.2f} GB"
    except Exception as e:
        return f"[{tag}] xpu-smi error (device {device_id}): {e}"

def get_cuda_memory_used_from_nvidia_smi(tag, device_id=0):
    if device_type != "cuda":
        return
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader", "-i", str(device_id)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        used_mem_mb = float(result.stdout.strip())
        return f"[{tag}] nvidia-smi memory used (device {device_id}): {used_mem_mb / 1024:.2f} GB"
    except Exception as e:
        return f"[{tag}] nvidia-smi error (device {device_id}): {e}"

# ----------------------------
# Main Training Logic
# ----------------------------
def main():
    rank, device = setup_distributed()

    # 2-layer test model
    model = nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)

    fully_shard(model)

    # Dummy dataset: (100-dim input, 10-class classification)
    X = torch.randn(100, 100)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(X, y)

    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=8)

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

            # Memory Logging
            memory_log = (
                get_xpu_memory_used_from_xpu_smi(f"Epoch {epoch}, Step {step}")
                if device_type == "xpu"
                else get_cuda_memory_used_from_nvidia_smi(f"Epoch {epoch}, Step {step}")
            )
            print(f"[Rank {rank}] Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}, {memory_log}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
