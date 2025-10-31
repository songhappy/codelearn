# Data (x, target): Created on every rank, but only rank 0’s x is fed into the pipeline, 
# and only the last rank’s target is used to compute the loss. Middle ranks just pass activations along.
# Global batch = 512
# World size = 4 GPUs
# → Mini-batch per GPU = 128
# → Micro-batches = 8
# → Micro-batch size = 128 / 8 = 16

import tempfile

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.pipelining import pipeline, SplitPoint
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
d_hid = 8
batch_size = 128
chunks = 8
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

def dw_g(rank, world_size, device):
    full_mod = MultiMLP(d_hid, n_layers=world_size)
    full_mod.to(device)
    stage_mod = full_mod.get_submodule(f"layers.{rank}")


    x = torch.randn(batch_size, d_hid, device=device)
    target = torch.randn(batch_size, d_hid, device=device)

    class CustomState:
        def __init__(self) -> None:
            self.i = 0

        def dw_builder(self):
            """This simulates a function attached to a model with a custom backward.
            Each call to builder gives a new dw_runner that has some updated state to compute the latest dw.
            """

            def dw_runner():
                # This inner function would be called by PipelineStage during `backward_weight_one_chunk`
                print(f"dw called {self.i}th time")
                self.i += 1

            return dw_runner

    cs = CustomState()

    stage = PipelineStage(
        stage_mod,
        rank,
        world_size,
        device,
        dw_builder=cs.dw_builder,
    )

    # Attach to a schedule
    schedule = ScheduleGPipe(
        stage, chunks, loss_fn=torch.nn.MSELoss(reduction="sum")
    )
    dist.barrier()

    # Run
    def _run_step(x):
        dist.barrier()
        if rank == 0:
            return schedule.step(x)
        elif rank == world_size - 1:
            return schedule.step(target=target)
        else:
            return schedule.step()

    out = _run_step(x)
    dist.barrier()

    print(x)
    print(out)


    print("---------")
    print(rank, cs.i, chunks)
    assert cs.i == chunks, f"dw_builder ran {cs.i} times, expected {chunks}"

    # Last rank checks result
    if rank == world_size - 1:
        ref_out = full_mod(x)
        print(ref_out)
        torch.testing.assert_close(out, ref_out)
    dist.barrier()

def run(rank, world_size):
    # Initialize the process group
    if torch.cuda.is_available():
        backend = "nccl"
        device = torch.device(f"cuda:{rank}")
    if torch.xpu.is_available():
        backend = "xccl"
        torch.xpu.set_device(rank)  # pin device
        device = torch.device(f"xpu:{rank}")
    dist.init_process_group(
        backend=backend,  # use "nccl" if CUDA instead of XPU
        init_method="tcp://127.0.0.1:29500",
        rank=rank,
        world_size=world_size
    )

    dw_g(rank, world_size, device)

    dist.destroy_process_group()


def main():
    world_size = 4
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
    
if __name__ == "__main__":
    main()
