#!/usr/bin/env python3
import os
import math
import sys
from typing import Tuple, Optional

import torch
import torch.distributed as dist

# Optional (recommended) for Intel XPU
try:
    import torch.xpu as xpu
    HAS_XPU = True
except Exception:
    HAS_XPU = False

def set_device_from_env(backend: str) -> int:
    """Bind this rank to the right local device for the chosen backend."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if backend == "xccl" and HAS_XPU:
        xpu.set_device(local_rank)
    elif backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    # gloo: keep CPU; no device binding required
    return local_rank

def choose_backend_from_env() -> str:
    """Pick a sensible default but respect DIST_BACKEND if provided."""
    env_backend = os.environ.get("DIST_BACKEND")
    if env_backend:
        return env_backend.lower()

    # Autoselect default
    if HAS_XPU:
        return "xccl"
    if torch.cuda.is_available():
        return "nccl"
    return "gloo"

def init_dist() -> str:
    """Initialize process group with fallbacks if needed."""
    preferred = choose_backend_from_env()
    tried = []
    for backend in ([preferred] +
                    [b for b in ("xccl", "nccl", "gloo") if b != preferred]):
        try:
            dist.init_process_group(backend=backend)
            return backend
        except Exception as e:
            tried.append((backend, str(e)))
    # If we got here, all failed:
    msgs = "\n".join([f"- {b}: {m}" for b, m in tried])
    raise RuntimeError(f"Failed to initialize any backend. Tried:\n{msgs}")

def tensor_device(backend: str) -> torch.device:
    """Match device to backend capabilities."""
    if backend == "xccl":
        return torch.device("xpu")
    if backend == "nccl":
        return torch.device("cuda")
    return torch.device("cpu")  # gloo

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

def pretty_pass_fail(ok: Optional[bool]) -> str:
    if ok is None:
        return "⚠️ SKIPPED"
    return "✅ PASS" if ok else "❌ FAIL"

def try_all_reduce(t: torch.Tensor,
                   op: dist.ReduceOp,
                   emulate_avg_if_needed: bool = False) -> bool:
    """
    Attempt all_reduce with given op.
    Returns True if native op used, False if emulated (only for AVG), and raises if not supported and not emulatable.
    For non-AVG ops, if unsupported, we re-raise to let caller decide to skip.
    """
    if op == dist.ReduceOp.AVG:
        try:
            dist.all_reduce(t, op=op)
            return True  # native AVG used
        except Exception:
            # emulate AVG via SUM/world_size
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= dist.get_world_size()
            return False
    else:
        # Non-AVG: just try, let caller catch
        dist.all_reduce(t, op=op)
        return True

def main():
    backend = init_dist()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Bind device after we know the backend
    local_rank = set_device_from_env(backend)
    dev = tensor_device(backend)

    if rank == 0:
        print(f"\n=== all_reduce reproducible test (backend={backend}, world_size={world_size}) ===\n")

    # FLOAT ops tensors
    float_val = float(rank + 1)
    t_sum  = torch.tensor([float_val], device=dev, dtype=torch.float32)
    t_avg  = torch.tensor([float_val], device=dev, dtype=torch.float32)
    t_prod = torch.tensor([float_val], device=dev, dtype=torch.float32)
    t_min  = torch.tensor([float_val], device=dev, dtype=torch.float32)
    t_max  = torch.tensor([float_val], device=dev, dtype=torch.float32)

    # INT ops tensors (bitwise require integer dtype)
    int_val = int(rank + 1)
    t_band = torch.tensor([int_val], device=dev, dtype=torch.int32)
    t_bor  = torch.tensor([int_val], device=dev, dtype=torch.int32)
    t_bxor = torch.tensor([int_val], device=dev, dtype=torch.int32)

    # Print initial values (per-rank)
    print(f"[rank {rank}] init float={float_val:.1f} int={int_val}")

    dist.barrier()

    # ---- SUM ----
    try_all_reduce(t_sum, dist.ReduceOp.SUM)
    dist.barrier()

    # ---- AVG ---- (native if available, else emulate SUM/world_size)
    try:
        used_native_avg = try_all_reduce(t_avg, dist.ReduceOp.AVG, emulate_avg_if_needed=True)
    except Exception as e:
        # As a last resort (shouldn't happen), emulate AVG
        dist.all_reduce(t_avg, op=dist.ReduceOp.SUM)
        t_avg /= world_size
        used_native_avg = False
    dist.barrier()

    # ---- PRODUCT ----
    try:
        try_all_reduce(t_prod, dist.ReduceOp.PRODUCT)
        prod_supported = True
    except Exception:
        prod_supported = False
    dist.barrier()

    # ---- MIN ----
    try:
        try_all_reduce(t_min, dist.ReduceOp.MIN)
        min_supported = True
    except Exception:
        min_supported = False
    dist.barrier()

    # ---- MAX ----
    try:
        try_all_reduce(t_max, dist.ReduceOp.MAX)
        max_supported = True
    except Exception:
        max_supported = False
    dist.barrier()

    # ---- BAND ----
    try:
        try_all_reduce(t_band, dist.ReduceOp.BAND)
        band_supported = True
    except Exception:
        band_supported = False
    dist.barrier()

    # ---- BOR ----
    try:
        try_all_reduce(t_bor, dist.ReduceOp.BOR)
        bor_supported = True
    except Exception:
        bor_supported = False
    dist.barrier()

    # ---- BXOR ----
    try:
        try_all_reduce(t_bxor, dist.ReduceOp.BXOR)
        bxor_supported = True
    except Exception:
        bxor_supported = False
    dist.barrier()

    # Expected results
    sum_exp, avg_exp, prod_exp, max_exp = expected_for_float_ops(world_size)
    min_exp = 1.0
    band_exp, bor_exp, bxor_exp = expected_for_bitwise_ops(world_size)

    # Helper: gather a scalar from all ranks
    def gather_scalar(t):
        gl = [torch.empty_like(t) for _ in range(world_size)]
        dist.all_gather(gl, t)
        return [v.item() for v in gl]

    # Gather results
    sum_all  = gather_scalar(t_sum)
    avg_all  = gather_scalar(t_avg)
    prod_all = gather_scalar(t_prod)  if prod_supported else None
    min_all  = gather_scalar(t_min)   if min_supported  else None
    max_all  = gather_scalar(t_max)   if max_supported  else None
    band_all = gather_scalar(t_band)  if band_supported else None
    bor_all  = gather_scalar(t_bor)   if bor_supported  else None
    bxor_all = gather_scalar(t_bxor)  if bxor_supported else None

    if rank == 0:
        def allclose(vals, tgt, rtol=1e-5, atol=1e-6):
            return all(abs(v - tgt) <= (atol + rtol * abs(tgt)) for v in vals)

        print("\n--- Results ---")
        ok_sum  = allclose(sum_all,  sum_exp)
        ok_avg  = allclose(avg_all,  avg_exp)
        ok_prod = allclose(prod_all, prod_exp, rtol=1e-5, atol=1e-5) if prod_all is not None else None
        ok_min  = allclose(min_all,  min_exp)                           if min_all  is not None else None
        ok_max  = allclose(max_all,  max_exp)                           if max_all  is not None else None
        ok_band = all(v == band_exp for v in band_all)                  if band_all is not None else None
        ok_bor  = all(v == bor_exp  for v in bor_all)                   if bor_all  is not None else None
        ok_bxor = all(v == bxor_exp for v in bxor_all)                  if bxor_all is not None else None

        print(f"SUM:    got={sum_all[0]}  exp={sum_exp}   {pretty_pass_fail(ok_sum)}")
        print(f"AVG:    got={avg_all[0]}  exp={avg_exp}   {pretty_pass_fail(ok_avg)}"
              + (" (native AVG)" if used_native_avg else " (emulated via SUM/world_size)"))
        print(f"PROD:   {'got='+str(prod_all[0]) if prod_all else 'n/a'}  exp={prod_exp}  {pretty_pass_fail(ok_prod)}")
        print(f"MIN:    {'got='+str(min_all[0])  if min_all  else 'n/a'}  exp={min_exp}   {pretty_pass_fail(ok_min)}")
        print(f"MAX:    {'got='+str(max_all[0])  if max_all  else 'n/a'}  exp={max_exp}   {pretty_pass_fail(ok_max)}")
        print(f"BAND:   {'got='+str(band_all[0]) if band_all else 'n/a'}  exp={band_exp}  {pretty_pass_fail(ok_band)}")
        print(f"BOR:    {'got='+str(bor_all[0])  if bor_all  else 'n/a'}  exp={bor_exp}   {pretty_pass_fail(ok_bor)}")
        print(f"BXOR:   {'got='+str(bxor_all[0]) if bxor_all else 'n/a'}  exp={bxor_exp}  {pretty_pass_fail(ok_bxor)}")

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
