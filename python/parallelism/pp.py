# Minimal 2-stage pipeline parallelism (PyTorch pipelining APIs)
# Run: torchrun --nproc_per_node 2 pp.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage0 = nn.Sequential(nn.Linear(1024, 128), nn.ReLU())
        self.stage1 = nn.Linear(128, 256)  # logits

    def forward(self, x):
        x = self.stage0(x)
        return self.stage1(x)

def make_dataloader(device, batch_size=32, total_samples=1024):
    x = torch.randn(total_samples, 1024, device=device)
    y = torch.randint(0, 256, (total_samples,), device=device)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False, drop_last=False)

def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert world_size == 2, "This minimal example assumes exactly 2 ranks (2 pipeline stages)."

    # Backend / device init (CUDA -> NCCL; otherwise XPU -> XCCL)
    backend = "nccl" if torch.cuda.is_available() else "xccl"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    device_type = "cuda" if torch.cuda.is_available() else "xpu"
    device = torch.device(device_type, local_rank)
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
    else:
        if hasattr(torch, "xpu") and hasattr(torch.xpu, "set_device"):
            torch.xpu.set_device(local_rank)

    torch.manual_seed(0)

    # Build the full model on each rank
    model = ToyModel().to(device)

    # Declare split & capture shapes with an example micro-batch
    # Split BEFORE stage1 => rank0 runs stage0; rank1 runs stage1
    example_mb = torch.randn(8, 1024, device=device)
    pipe = pipeline(
        module=model,
        mb_args=(example_mb,),
        split_spec={"stage1": SplitPoint.BEGINNING},
    )

    # Build local stage runtime (use positional args for wider compatibility)
    stage = pipe.build_stage(rank, device, dist.group.WORLD)

    # Only the INPUT stage (rank 0) computes loss; others pass None
    loss_fn = nn.CrossEntropyLoss() if rank == 0 else None
    schedule = ScheduleGPipe(stage=stage, n_microbatches=4, loss_fn=loss_fn)

    # Optimizer over the original model (portable across wheels)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    # Dataloader only on rank0; rank1 matches the step count to avoid hangs
    EPOCHS = 2
    BATCH_SIZE = 32
    TOTAL_SAMPLES = 1024

    if rank == 0:
        loader = make_dataloader(device, batch_size=BATCH_SIZE, total_samples=TOTAL_SAMPLES)
        steps = len(loader)
        steps_tensor = torch.tensor([steps], dtype=torch.int64, device=device)
        dist.broadcast(steps_tensor, src=0)
    else:
        steps_tensor = torch.tensor([0], dtype=torch.int64, device=device)
        dist.broadcast(steps_tensor, src=0)
        steps = int(steps_tensor.item())

    for epoch in range(EPOCHS):
        if rank == 0:
            for xb, yb in loader:
                opt.zero_grad(set_to_none=True)
                # Input stage provides inputs + targets
                schedule.step(xb, target=yb)
                opt.step()
            print(f"[Epoch {epoch+1}] rank0 done")
        else:
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                # Non-input stage participates without inputs/targets
                schedule.step()
                opt.step()

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
