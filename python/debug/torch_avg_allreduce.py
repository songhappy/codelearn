import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, world_size):
    # Initialize the process group
    dist.init_process_group(
        init_method="tcp://127.0.0.1:29500",
        rank=rank,
        backend="xccl",       # or "gloo" if no GPU
        world_size=world_size
    )

    # Each process starts with a different tensor
    tensor = torch.tensor([rank + 1.0], device="xpu")
    print(f"Before all_reduce, rank {rank} has {tensor.item()}")

    # All-reduce with sum
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)

    # Average by dividing by world size
    tensor /= world_size
    print(f"After all_reduce(avg), rank {rank} has {tensor.item()}")

    dist.destroy_process_group()


def main():
    world_size = 2   # try 2 processes
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
