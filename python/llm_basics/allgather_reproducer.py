import torch
import torch.distributed as dist
import argparse
import time
import os

def get_device_and_backend():
    if torch.backends.mps.is_available():
        return "mps", "gloo"  # Apple fallback
    elif torch.cuda.is_available():
        return "cuda", "nccl"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu", "xccl"  # Assumes PyTorch built with XPU backend (e.g., ccl or xccl)
    else:
        raise RuntimeError("No supported device found (CUDA or XPU required).")

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
    else:
        raise ValueError("Unsupported device type")

    return device, device_type, sync_fn

def cleanup():
    dist.destroy_process_group()

def main(print_each=False):
    device, device_type, sync_fn = setup()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Allocate ~2 GiB float32 tensor per rank
    input_tensor = torch.ones(536_870_912, device=device) * rank
    gathered = [torch.zeros_like(input_tensor) for _ in range(world_size)]

    sync_fn()
    start_time = time.time()

    for i in range(1000):
        dist.all_gather(gathered, input_tensor)
        sync_fn()
        if print_each and i % 100 == 0:
            print(f"[{rank}] Completed iteration {i}")

    sync_fn()
    end_time = time.time()
    total_time = end_time - start_time

    if rank == 0:
        print(f"\n[{rank}] Total time for 1000 all_gather calls: {total_time:.3f} seconds")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Print every 100 iterations")
    args = parser.parse_args()

    main(print_each=args.verbose)
