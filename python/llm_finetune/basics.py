import os
import torch
import torch.distributed as dist

def setup(rank, world_size):
    dist.init_process_group(
        backend="xccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    torch.xpu.set_device(rank)

def cleanup():
    dist.destroy_process_group()


def all_gather_example(rank, world_size):
    input_tensor = torch.tensor([rank], dtype=torch.int64, device=rank)
    gather_list = [torch.zeros(1, dtype=torch.int64, device=rank) for _ in range(world_size)]
    dist.all_gather(gather_list, input_tensor)
    print(f"[{rank}] all_gather: {gather_list}")


def all_reduce_example(rank, world_size):
    tensor = torch.tensor([rank + 1.0], device=rank)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"[{rank}] all_reduce result: {tensor}")

def broadcast_example(rank, world_size):
    tensor = torch.zeros(1, device=rank)
    if rank == 0:
        tensor = torch.tensor([123.0], device=rank)
    dist.broadcast(tensor, src=0)
    print(f"[{rank}] after broadcast: {tensor}")

def reduce_example(rank, world_size):
    tensor = torch.tensor([rank + 1.0], device=rank)
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f"[{rank}] reduced sum: {tensor}")

def reduce_scatter_example(rank, world_size):
    input_list = [torch.ones(2, device=rank) * (rank + 1) for _ in range(world_size)]
    print(input_list)
    output = torch.zeros(2, device=rank)
    dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM)
    print(f"[{rank}] reduce_scatter output: {output}")

def p2p_send_recv_example(rank, world_size):
    tensor = torch.tensor([rank + 10], device=rank)
    if rank == 0:
        dist.send(tensor, dst=1)
        print(f"[{rank}] sent: {tensor}")
    elif rank == 1:
        recv_tensor = torch.zeros(1, device=rank)
        dist.recv(recv_tensor, src=0)
        print(f"[{rank}] received: {recv_tensor}")

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])


    setup(rank, world_size)

    # Run each NCCL operation
    all_gather_example(rank, world_size)
    all_reduce_example(rank, world_size)
    broadcast_example(rank, world_size)
    reduce_example(rank, world_size)
    reduce_scatter_example(rank, world_size)

    # Only run point-to-point if world_size >= 2
    if world_size >= 2:
        p2p_send_recv_example(rank, world_size)

    cleanup()
