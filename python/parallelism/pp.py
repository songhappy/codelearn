# python pp.py
# Minimal pipeline parallel demo: 2 stages, 2 micro-batches, CPU+Gloo.

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

WORLD_SIZE = 2     # 2 stages (ranks 0 and 1)
D = 4              # feature dim
BATCH = 4          # mini-batch per step
CHUNKS = 2         # micro-batches -> micro-batch size = 2

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
    def forward(self, x): return self.net(x)

def run(rank, world_size):
    dist.init_process_group(
        backend="xccl", init_method="tcp://127.0.0.1:29500",
        rank=rank, world_size=world_size
    )
    device = torch.device("xpu")

    # Each rank holds one stage of the model (stage0 -> stage1).
    torch.manual_seed(0)                          # deterministic init on all ranks
    stage_mod = Block(D).to(device)               # one block per stage

    # Build the pipeline stage and schedule.
    stage = PipelineStage(stage_mod, rank, world_size, device)
    schedule = ScheduleGPipe(stage, CHUNKS, loss_fn=nn.MSELoss(reduction="sum"))

    # Prepare input/target: rank0 feeds x; last rank feeds target for loss.
    x = torch.randn(BATCH, D) if rank == 0 else None
    target = torch.randn(BATCH, D) if rank == world_size - 1 else None

    dist.barrier()
    # One pipelined training step:
    if rank == 0:
        schedule.step(x)                   # feed input
    elif rank == world_size - 1:
        schedule.step(target=target)       # provide target for loss/backward
    else:
        schedule.step()                    # middle stages just run
    dist.barrier()

    dist.destroy_process_group()

def main():
    mp.spawn(run, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)

if __name__ == "__main__":
    main()
