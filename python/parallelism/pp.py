# python pp.py
# Minimal pipeline parallel demo: 2 stages, 2 micro-batches, CPU+Gloo.

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from torch.distributed.pipelining import pipeline, SplitPoint

WORLD_SIZE = 2     # 2 stages (ranks 0 and 1)
D = 4              # feature dim
BATCH = 4          # mini-batch per step
CHUNKS = 2         # micro-batches -> micro-batch size = 2
torch.manual_seed(0)

class MLPModule(torch.nn.Module):
    def __init__(self, d_hid: int):
        super().__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x
  
class MultiMLP(torch.nn.Module):
    def __init__(self, d_hid: int, n_layers: int = 2):
        super().__init__()
        self.layers = torch.nn.ModuleList([MLPModule(d_hid) for _ in range(n_layers)])
        # For testing purpose only, this should be defined by user
        self.split_spec = {
            f"layers.{i}": SplitPoint.BEGINNING for i in range(1, n_layers)
        }

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def run(rank, world_size):
    if torch.cuda.is_available():
        backend = "nccl"
        device = torch.device(f"cuda:{rank}")
    if torch.xpu.is_available():
        backend = "xccl"
        device = torch.device(f"xpu:{rank}")
    dist.init_process_group(
        backend=backend, init_method="tcp://127.0.0.1:29500",
        rank=rank, world_size=world_size
    )

    full_mod = MultiMLP(d_hid=D, n_layers=world_size)
    full_mod.to(device)

    # Each rank holds one stage of the model (stage0 -> stage1).
    stage_mod = full_mod.get_submodule(f"layers.{rank}")            # one block per stage

    # Build the pipeline stage and schedule.
    stage = PipelineStage(stage_mod, rank, world_size, device)
    schedule = ScheduleGPipe(stage, CHUNKS, loss_fn=nn.MSELoss(reduction="sum"))

    # Prepare input/target: rank0 feeds x; last rank feeds target for loss.
    x = torch.randn(BATCH, D, device=device) 
    target = torch.randn(BATCH, D, device=device) 

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
