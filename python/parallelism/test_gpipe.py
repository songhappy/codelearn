#  torchrun --nproc_per_node 2 test_gpipe.py
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

def send_receive(rank, device):
     # --- direct send/recv test ---
    print(f"[rank {rank}] testing send/recv")
    dist.barrier()
    if rank == 0:
        t = torch.ones(8, device=device)
        dist.send(t, dst=1)
    elif rank == 1:
        t = torch.empty(8, device=device)
        dist.recv(t, src=0)
        print(f"[rank {rank}] received tensor: {t}")
    dist.barrier()

    if rank == 0:
        print("✅ send/recv test done")

def simple_g(rank, world_size, device):
        # --- tiny 2-stage model ---
    model = nn.Sequential(
        nn.Linear(1024, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
    ).to(device)

    # --- build pipeline ---
    example = torch.randn(4, 1024, device=device)
    pipe = pipeline(model, mb_args=(example,), split_spec={"2": SplitPoint.BEGINNING})
    stage = pipe.build_stage(rank, device, dist.group.WORLD)

    # --- sample input/target ---
    x = torch.randn(8, 1024, device=device)
    target = torch.randn(8, 256, device=device)

    # --- your lockstep schedule snippet ---
    schedule = ScheduleGPipe(stage=stage, n_microbatches=2,
                             loss_fn=(nn.MSELoss(reduction="sum") if rank == world_size - 1 else None))
    dist.barrier()  # after schedule build

    for it in range(3):
        dist.barrier()
        if rank == 0:
            out = schedule.step(x)                # supply inputs only on rank 0
        elif rank == world_size - 1:
            out = schedule.step(target=target)    # supply target only on last rank
        else:
            out = schedule.step()
        dist.barrier()
    # ----------------------------------------------------

def main():
    # --- setup for XPU ---
    backend = "xccl"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.xpu.set_device(local_rank)
    device = torch.device("xpu", local_rank)
    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    simple_g(rank, world_size, device)
    send_receive(rank, device)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
