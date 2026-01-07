
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, world_size):
    # Initialize the process group
    dist.init_process_group(
        backend="xccl",       # or "gloo" if no GPU
        init_method="tcp://127.0.0.1:29500",
        rank=rank,
        world_size=world_size
    )
    to_reduce_scatter = [torch.ones(3, 3) * rank for rank in range(world_size)]
    output_tensor = torch.empty(3, 3)

    dist.reduce_scatter(output_tensor, to_reduce_scatter)
    expected_tensor = torch.ones(3, 3) * dist.get_rank() * world_size
    print(output_tensor)
    print(expected_tensor)

    output_tensor = torch.empty(3, 3)
    dist.reduce_scatter(output_tensor, to_reduce_scatter, op=dist.ReduceOp.AVG)
    expected_tensor = torch.ones(3, 3) * dist.get_rank()
    print(output_tensor)
    print(expected_tensor)
    dist.destroy_process_group()


def main():
    world_size = 2   # try 2 processes
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
    # dist.barrier()


if __name__ == "__main__":
    main()

