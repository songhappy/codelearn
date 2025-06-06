import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from torchvision import models
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def setup_distributed(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("xpu:xccl", rank=rank, world_size=world_size)
    torch.xpu.set_device(rank)
    return torch.device(f"xpu:{rank}")


def train(rank, world_size, distributed):
    if distributed:
        device = setup_distributed(rank, world_size)
    else:
        device = torch.device("xpu:0")
        torch.xpu.set_device(device)

    # Initialize model and wrap with FSDP
    model = models.resnet50().to(device)
    model.train()
    model = FSDP(model)  # Use FSDP to shard the whole model safely

    # Dummy input and target
    input_data = torch.randn(32, 3, 224, 224, device=device)
    target = torch.randint(0, 1000, (32,), device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Profiler setup
    prof = profile(
        activities=[ProfilerActivity.XPU,
                    ProfilerActivity.CPU,
                    ],
        on_trace_ready=tensorboard_trace_handler(f'./log/rank{rank}'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    )

    prof.start()
    for idx in range(20):
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        torch.xpu.synchronize()
        if idx == 6:
            prof.step()
    try:
        prof.stop()
    except Exception as e:
        print(f"[Rank {rank}] profiler stop error (likely PTI not fully supported): {e}")

    if rank == 0:
        print(prof.key_averages().table(sort_by="xpu_time_total", row_limit=10))

    if distributed:
        dist.destroy_process_group()


def main():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        train(rank, world_size, distributed=True)
    else:
        print("Running in single-XPU mode")
        train(rank=0, world_size=1, distributed=False)


if __name__ == "__main__":
    main()
