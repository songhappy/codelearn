

# export ZE_AFFINITY_MASK=0,1
# torchrun --nproc_per_node 8 test_allreduce.py

import os
import torch
import torch.distributed as dist

def main():
    # --- setup for XPU ---
    backend = "xccl"
    device = torch.device("xpu", int(os.environ.get("LOCAL_RANK", 0)))
    torch.xpu.set_device(device)
    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # --- sanity comm test (before building pipeline) ---
    x = torch.ones(4, device=device)
    dist.all_reduce(x)
    if rank == 0:
        print(f"sanity all_reduce: {x}")  # expect tensor([world_size, world_size, world_size, world_size])
    dist.barrier()
    # ----------------------------------------------------

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
