import torch
import torch.distributed as dist
import argparse
import time
import os

def get_device_and_backend():
    if torch.backends.mps.is_available():
        return "mps", "gloo"
    elif torch.cuda.is_available():
        return "cuda", "nccl"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu", "xccl"  # Replace with "gloo" if your torch doesn't support "xccl"
    else:
        raise RuntimeError("No supported device found.")

def setup():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    device_type, backend = get_device_and_backend()
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if device_type == "cuda":
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(local_rank)
        sync_fn = torch.cuda.synchronize
    elif device_type == "xpu":
        device = torch.device("xpu", local_rank)
        torch.xpu.set_device(local_rank)
        sync_fn = torch.xpu.synchronize
    elif device_type == "mps":
        device = torch.device("mps")
        sync_fn = lambda: None
    else:
        raise ValueError("Unsupported device type")

    return device, device_type, sync_fn

def cleanup():
    dist.destroy_process_group()

def benchmark_allgather(device, sync_fn, world_size, rank, print_each=False):
    input_tensor = torch.ones(536_870_912, device=device) * rank
    gathered = [torch.zeros_like(input_tensor) for _ in range(world_size)]

    sync_fn()
    start = time.time()
    for i in range(1000):
        dist.all_gather(gathered, input_tensor)
    sync_fn()
    end = time.time()

    return end - start

def benchmark_reducescatter(device, sync_fn, world_size, rank, print_each=False):
    scatter_input = [torch.ones(536_870_912, device=device) * (rank + i) for i in range(world_size)]
    output_tensor = torch.zeros_like(scatter_input[0])

    sync_fn()
    start = time.time()
    for i in range(1000):
        dist.reduce_scatter(output_tensor, scatter_input, op=dist.ReduceOp.SUM)
    sync_fn()
    end = time.time()

    return end - start

def main(print_each=False):
    device, device_type, sync_fn = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # add warmup iterations
    for _ in range(10):
        dist.all_gather([torch.zeros(536_870_912, device=device) for _ in range(world_size)], torch.ones(536_870_912, device=device))
        dist.reduce_scatter(torch.zeros(536_870_912, device=device), [torch.ones(536_870_912, device=device) * i for i in range(world_size)], op=dist.ReduceOp.SUM)

    allgather_time = benchmark_allgather(device, sync_fn, world_size, rank, print_each)
    reducescatter_time = benchmark_reducescatter(device, sync_fn, world_size, rank, print_each)

    if rank == 0:
        print(f"\n[{rank}] Total time for 1000 all_gather calls:     {allgather_time:.3f} seconds")
        print(f"[{rank}] Total time for 1000 reduce_scatter calls: {reducescatter_time:.3f} seconds")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Print every 100 iterations")
    args = parser.parse_args()

    main(print_each=args.verbose)
