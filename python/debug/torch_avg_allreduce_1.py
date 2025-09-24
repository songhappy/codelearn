import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def main():
    # Initialize the process group
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="xccl", rank=rank, world_size=world_size)

    # Each process starts with a different tensor
    tensor = torch.tensor([rank + 1.0], device="xpu")
    print(f"Before all_reduce, rank {rank} has {tensor.item()}")

    # All-reduce with sum
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)

    # Average by dividing by world size
    tensor /= world_size
    print(f"After all_reduce(avg), rank {rank} has {tensor.item()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
