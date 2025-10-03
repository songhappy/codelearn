import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Optional: Intel XPU
try:
    import torch.xpu as xpu
    HAS_XPU_MODULE = True
except Exception:
    HAS_XPU_MODULE = False

def xpu_available():
    return HAS_XPU_MODULE and getattr(xpu, "is_available", lambda: False)() and xpu.device_count() > 0

def cuda_available():
    return torch.cuda.is_available() and torch.cuda.device_count() > 0

def pick_backend_and_kind():
    """
    Decide backend ('xccl'|'nccl'|'gloo') and device kind ('xpu'|'cuda'|'cpu').
    Priority: XPU→CUDA→CPU; respect DIST_BACKEND if valid.
    """
    env_backend = os.environ.get("DIST_BACKEND", "").lower().strip()
    if env_backend in ("xccl", "nccl", "gloo"):
        kind = {"xccl": "xpu", "nccl": "cuda", "gloo": "cpu"}[env_backend]
        return env_backend, kind

    if xpu_available():
        return "xccl", "xpu"
    if cuda_available():
        return "nccl", "cuda"
    return "gloo", "cpu"

def set_local_device(kind: str, local_rank: int):
    if kind == "xpu":
        count = xpu.device_count()
        if local_rank >= count:
            raise RuntimeError(f"LOCAL_RANK {local_rank} >= XPU device_count {count}")
        xpu.set_device(local_rank)
    elif kind == "cuda":
        count = torch.cuda.device_count()
        if local_rank >= count:
            raise RuntimeError(f"LOCAL_RANK {local_rank} >= CUDA device_count {count}")
        torch.cuda.set_device(local_rank)
    # cpu/gloo: no device binding

def device_string(kind: str, local_rank: int) -> str:
    return f"{kind}:{local_rank}" if kind in ("xpu", "cuda") else "cpu"

def init_pg(backend: str, rank: int, world_size: int, init_method: str):
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )

def run(rank, world_size, init_method):
    backend, kind = pick_backend_and_kind()

    # single-node spawn: local_rank == rank
    local_rank = rank

    # Guard against “imported but not usable” scenarios
    if kind == "xpu" and not xpu_available():
        # Fallback automatically if user didn't force xccl
        if os.environ.get("DIST_BACKEND", "").lower().strip() == "xccl":
            raise RuntimeError("DIST_BACKEND=xccl but no usable XPU device found.")
        backend, kind = ("nccl", "cuda") if cuda_available() else ("gloo", "cpu")

    if kind == "cuda" and not cuda_available():
        if os.environ.get("DIST_BACKEND", "").lower().strip() == "nccl":
            raise RuntimeError("DIST_BACKEND=nccl but no usable CUDA device found.")
        backend, kind = ("gloo", "cpu")

    set_local_device(kind, local_rank)
    init_pg(backend, rank, world_size, init_method)

    dev = device_string(kind, local_rank)

    # Each process starts with a different tensor
    tensor = torch.tensor([float(rank + 1)], device=dev, dtype=torch.float32)
    if rank == 0:
        print(f"\n=== backend={backend}, device_kind={kind}, world_size={world_size} ===\n")
    print(f"[rank {rank}] before all_reduce: {tensor.item()} on {dev}")

    # Average robustly: SUM then divide (works on all backends)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size

    print(f"[rank {rank}] after  all_reduce(avg): {tensor.item()} on {dev}")

    dist.destroy_process_group()

def main():
    world_size = int(os.environ.get("WORLD_SIZE_OVERRIDE", 2))  # tweak for quick tests
    init_method = os.environ.get("DIST_INIT", "tcp://127.0.0.1:29500")
    mp.spawn(run, args=(world_size, init_method), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
