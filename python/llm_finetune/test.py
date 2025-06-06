import torch
from torch import distributed as dist
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, ShardingStrategy)
from torch.nn import Linear, Module, Conv1d
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD
from torch.testing._internal.common_fsdp import get_full_params
import os
import socket

os.environ['RANK'] = str(os.environ.get('PMIX_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PALS_LOCAL_SIZE', 1))
os.environ['MASTER_ADDR'] = socket.gethostname()  # your master address
os.environ['MASTER_PORT'] = '29500'  # your master port

RANK=int(os.environ['RANK'])
WORLD_SIZE=int(os.environ['WORLD_SIZE'])
MANUAL_SEED=5
device_type = torch.device(f"xpu:{RANK}")
LINEAR_DIM=1
WITH_BIAS=True
DATATYPE=torch.float32
LR=0.1

class Model(Module):
    def __init__(self, wrap_fsdp):
        super().__init__()
        torch.manual_seed(MANUAL_SEED)
        self.inner = Linear(LINEAR_DIM, LINEAR_DIM, bias=WITH_BIAS, dtype=DATATYPE)

    def forward(self, x):
        return self.inner(x)

def train(wrap_fsdp):
    torch.manual_seed(MANUAL_SEED)
    model = Model(wrap_fsdp).to(device_type)
    if wrap_fsdp:
        model = FSDP(model, device_id=device_type, limit_all_gathers=False)
    else:
        model = DistributedDataParallel(model, device_ids=[device_type])
    optim = SGD(model.parameters(), lr=LR)
    in_data= torch.rand(LINEAR_DIM, LINEAR_DIM, device=device_type, dtype=DATATYPE)
    in_data.requires_grad = True
    for _ in range(1):
        out = model(in_data)
        loss = out.sum()
        loss.backward()
        optim.step()
        optim.zero_grad()
    if wrap_fsdp:
        return get_full_params(model)
    return list(model.parameters())

torch.xpu.set_device(device_type)
dist.init_process_group("xccl")
print(f"dist init r={RANK}, world={WORLD_SIZE}")

# DDP
ddp_state = train(wrap_fsdp=False)

# FSDP
fsdp_state = train(wrap_fsdp=True)

#print(f"DONE -  ddp_state = {ddp_state}")
#print(f"DONE - fsdp_state = {fsdp_state}")

if RANK == 0:
    for param1, param2 in zip(ddp_state, fsdp_state):
        if not torch.equal(param1, param2):
            print("Compare failed!!!!")
            print("ddp param:", param1, param1.grad)
            print("fsdp param:", param2, param2.grad)
            print(torch.abs(param1-param2))
            print("diff:", torch.max(torch.abs(param1-param2)))
        else:
            print("Compared passed!!!")
        #print("ddp param:", param1)
        #print("fsdp param:", param2)
        #print("diff:", torch.max(torch.abs(param1-param2)))

dist.destroy_process_group()
