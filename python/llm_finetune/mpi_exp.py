# torchrun --nproc_per_node=2 mpi_example.py
# # mpi_example.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
import os

# Initialize MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

def setup_distributed():
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='xccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main():
    setup_distributed()

    # Create a simple model
    model = nn.Linear(10, 1)
    torch.manual_seed(0)
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Create a dummy input and target
    input = torch.randn(10).to(rank)
    target = torch.randn(1).to(rank)

    # Define a loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Training loop
    for i in range(5):
        optimizer.zero_grad()
        output = ddp_model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        print(f"Rank {rank}, Step {i}, Loss: {loss.item()}")

    cleanup()

if __name__ == "__main__":
    main()
