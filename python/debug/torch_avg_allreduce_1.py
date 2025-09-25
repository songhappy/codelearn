#!/usr/bin/env python3
import os
import math
import sys
from typing import Tuple

import torch
import torch.distributed as dist

# Optional but recommended for XPU
try:
    import torch.xpu as xpu
    has_xpu = True
except Exception:
    has_xpu = False

def set_device_from_env():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if has_xpu:
        xpu.set_device(local_rank)
    else:
        # Fallback to CUDA/CPU if someone runs this without XPU
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        else:
            pass
    return local_rank

def init_dist():
    # torchrun sets RANK, WORLD_SIZE, LOCAL_RANK and the default env init method.
    backend = os.environ.get("DIST_BACKEND", "xccl")  # default to XCCL
    # If you want to pass rank/world_size explicitly, uncomment below:
    # rank = int(os.environ["RANK"]); world_size = int(os.environ["WORLD_SIZE"])
    # dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    dist.init_process_group(backend=backend)
    return backend

def tensor_device():
    # Prefer XPU → CUDA → CPU
    if has_xpu:
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def expected_for_float_ops(world_size: int) -> Tuple[float, float, float, float]:
    """Returns (sum, avg, product, minmax_max) given ranks have values 1..world_size."""
    s = world_size * (world_size + 1) / 2.0
    avg = s / world_size
    prod = math.prod(range(1, world_size + 1))
    maxv = float(world_size)
    return s, avg, float(prod), maxv  # min is always 1.0

def expected_for_bitwise_ops(world_size: int) -> Tuple[int, int, int]:
    """Reduce 1..world_size with BAND, BOR, BXOR."""
    vals = list(range(1, world_size + 1))
    b_and = vals[0]
    for v in vals[1:]:
        b_and &= v
    b_or = 0
    for v in vals:
        b_or |= v
    b_xor = 0
    for v in vals:
        b_xor ^= v
    return b_and, b_or, b_xor

def pretty_pass_fail(ok: bool) -> str:
    return "✅ PASS" if ok else "❌ FAIL"

def main():
    local_rank = set_device_from_env()
    backend = init_dist()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dev = tensor_device()

    if rank == 0:
        print(f"\n=== all_reduce reproducible test (backend={backend}, world_size={world_size}) ===\n")

    # FLOAT ops: SUM, AVG, PRODUCT, MIN, MAX
    float_val = float(rank + 1)
    t_sum = torch.tensor([float_val], device=dev, dtype=torch.float32)
    t_avg = torch.tensor([float_val], device=dev, dtype=torch.float32)
    t_prod = torch.tensor([float_val], device=dev, dtype=torch.float32)
    t_min = torch.tensor([float_val], device=dev, dtype=torch.float32)
    t_max = torch.tensor([float_val], device=dev, dtype=torch.float32)

    # INT ops: BAND, BOR, BXOR (require integer dtype)
    int_val = int(rank + 1)
    t_band = torch.tensor([int_val], device=dev, dtype=torch.int32)
    t_bor  = torch.tensor([int_val], device=dev, dtype=torch.int32)
    t_bxor = torch.tensor([int_val], device=dev, dtype=torch.int32)

    # Print initial values
    print(f"[rank {rank}] init float={float_val:.1f} int={int_val}")

    dist.barrier()

    # ---- SUM ----
    dist.all_reduce(t_sum, op=dist.ReduceOp.SUM)
    dist.barrier()

    # ---- AVG ---- (try native AVG, else emulate with SUM/world_size)
    avg_used_native = True
    # try:
    # except RuntimeError:
    #     print("====err AVG err ==============================================================")
    #     print(RuntimeError)
    #     # Backend may not support AVG; emulate: all-reduce SUM then divide.
    #     avg_used_native = False
    #     dist.all_reduce(t_avg, op=dist.ReduceOp.SUM)
    #     t_avg /= world_size
    
    dist.all_reduce(t_avg, op=dist.ReduceOp.AVG)
    dist.barrier()

    # ---- PRODUCT ----
    dist.all_reduce(t_prod, op=dist.ReduceOp.PRODUCT)
    dist.barrier()

    # ---- MIN ----
    dist.all_reduce(t_min, op=dist.ReduceOp.MIN)
    dist.barrier()

    # ---- MAX ----
    dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
    dist.barrier()

    # ---- BAND ----
    # dist.all_reduce(t_band, op=dist.ReduceOp.BAND)
    # dist.barrier()

    # ---- BOR ----
    dist.all_reduce(t_bor, op=dist.ReduceOp.BOR)
    dist.barrier()

    # ---- BXOR ----
    dist.all_reduce(t_bxor, op=dist.ReduceOp.BXOR)
    dist.barrier()

    # Compute expected results
    sum_exp, avg_exp, prod_exp, max_exp = expected_for_float_ops(world_size)
    min_exp = 1.0
    band_exp, bor_exp, bxor_exp = expected_for_bitwise_ops(world_size)

    # Gather results to rank 0 for clean verification/printing
    def gather_scalar(t):
        # all_gather into list of same dtype
        gather_list = [torch.empty_like(t) for _ in range(world_size)]
        dist.all_gather(gather_list, t)
        return [v.item() for v in gather_list]

    sum_all = gather_scalar(t_sum)
    avg_all = gather_scalar(t_avg)
    prod_all = gather_scalar(t_prod)
    min_all = gather_scalar(t_min)
    max_all = gather_scalar(t_max)
    band_all = gather_scalar(t_band)
    bor_all  = gather_scalar(t_bor)
    bxor_all = gather_scalar(t_bxor)

    if rank == 0:
        # All ranks should have identical post-reduce values for each op.
        def allclose(vals, tgt, rtol=1e-5, atol=1e-6):
            return all(abs(v - tgt) <= (atol + rtol * abs(tgt)) for v in vals)

        print("\n--- Results ---")
        ok_sum  = allclose(sum_all,  sum_exp)
        ok_avg  = allclose(avg_all,  avg_exp)
        ok_prod = allclose(prod_all, prod_exp, rtol=1e-5, atol=1e-5)
        ok_min  = allclose(min_all,  min_exp)
        ok_max  = allclose(max_all,  max_exp)
        ok_band = all(v == band_exp for v in band_all)
        ok_bor  = all(v == bor_exp  for v in bor_all)
        ok_bxor = all(v == bxor_exp for v in bxor_all)

        print(f"SUM:    got={sum_all[0]}  exp={sum_exp}   {pretty_pass_fail(ok_sum)}")
        print(f"AVG:    got={avg_all[0]}  exp={avg_exp}   {pretty_pass_fail(ok_avg)}"
              + (" (native AVG)" if avg_used_native else " (emulated via SUM/world_size)"))
        print(f"PROD:   got={prod_all[0]} exp={prod_exp}  {pretty_pass_fail(ok_prod)}")
        print(f"MIN:    got={min_all[0]}  exp={min_exp}   {pretty_pass_fail(ok_min)}")
        print(f"MAX:    got={max_all[0]}  exp={max_exp}   {pretty_pass_fail(ok_max)}")
        print(f"BAND:   got={band_all[0]} exp={band_exp}  {pretty_pass_fail(ok_band)}")
        print(f"BOR:    got={bor_all[0]}  exp={bor_exp}   {pretty_pass_fail(ok_bor)}")
        print(f"BXOR:   got={bxor_all[0]} exp={bxor_exp}  {pretty_pass_fail(ok_bxor)}")

    dist.barrier()
    if rank == 0:
        print("\nDone.\n")

    dist.destroy_process_group()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Make errors visible per-rank (useful when debugging multi-proc runs)
        print(f"[rank {os.environ.get('RANK','?')}] ERROR: {e}", file=sys.stderr)
        raise
