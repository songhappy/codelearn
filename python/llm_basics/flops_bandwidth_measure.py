import time
import torch

def _pick_device(device=None):
    if device is not None:
        return device
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("Neither XPU nor CUDA is available.")

def _backend_module(device):
    return torch.xpu if device == "xpu" else torch.cuda

def _time_it(devmod, iters, warmup, body, use_events=True):
    # Warmup
    for _ in range(warmup):
        body()
    devmod.synchronize()

    times_ms = []
    used_events = False

    if use_events and hasattr(devmod, "Event"):
        try:
            start, end = devmod.Event(enable_timing=True), devmod.Event(enable_timing=True)
            for _ in range(iters):
                start.record()
                body()
                end.record()
                devmod.synchronize()
                times_ms.append(start.elapsed_time(end))  # ms
            used_events = True
        except Exception:
            used_events = False

    if not used_events:
        for _ in range(iters):
            t0 = time.perf_counter()
            body()
            devmod.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1e3)  # ms
    return min(times_ms), ("events" if used_events else "wall-clock")

def gpu_gemm_tflops(
    M=8192, N=8192, K=8192,
    dtype=torch.bfloat16,
    iters=10, warmup=5,
    device=None, use_events=True,
):
    """
    GEMM throughput (TFLOPs) for A[M,K] @ B[K,N] on CUDA or XPU.
    """
    device = _pick_device(device)
    devmod = _backend_module(device)

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)

    ms, timing = _time_it(
        devmod, iters, warmup,
        body=lambda: (A @ B),
        use_events=use_events,
    )

    flops = 2.0 * M * N * K
    tflops = flops / (ms / 1e3) / 1e12
    print(
        f"[GEMM] Device: {device} | Best: {ms:.2f} ms ({timing}) | "
        f"{M}x{K} · {K}x{N} | ~{tflops:.2f} TFLOPs | dtype={dtype}"
    )
    return tflops

def gpu_mem_bandwidth(
    numel=64 * 1024 * 1024,  # 64M elements (~128 MiB per tensor with bf16)
    dtype=torch.bfloat16,
    iters=20, warmup=5,
    device=None, use_events=True,
    mode="copy",  # "copy" (D2D memcpy) or "triad" (y += x)
):
    """
    Measure effective device memory bandwidth on CUDA or XPU.

    mode:
      - "copy": dst.copy_(src) -> bytes moved = 2 * numel * element_size
      - "triad": y.add_(x)     -> bytes moved ≈ 3 * numel * element_size (read x, read y, write y)
    """
    device = _pick_device(device)
    devmod = _backend_module(device)

    # allocate
    y = torch.empty(numel, device=device, dtype=dtype)
    x = torch.empty_like(y)
    # Touch memory once outside of timing to avoid first-use effects
    x.fill_(1)
    y.zero_()
    elem_size = y.element_size()

    if mode == "copy":
        body = lambda: y.copy_(x)
        bytes_moved = 2.0 * numel * elem_size
    elif mode == "triad":
        # single-kernel fused add_: y += x
        body = lambda: y.add_(x)
        bytes_moved = 3.0 * numel * elem_size
    else:
        raise ValueError("mode must be 'copy' or 'triad'")

    ms, timing = _time_it(devmod, iters, warmup, body, use_events=use_events)

    gbps = bytes_moved / (ms / 1e3) / 1e9
    label = "D2D memcpy" if mode == "copy" else "Streaming triad (y += x)"
    print(
        f"[BW-{mode}] Device: {device} | Best: {ms:.2f} ms ({timing}) | "
        f"Elements: {numel:,} | ~{gbps:.2f} GB/s | dtype={dtype} | {label}"
    )
    return gbps

if __name__ == "__main__":
    # GEMM TFLOPs
    gpu_gemm_tflops()

    # Memory bandwidth (both styles)
    gpu_mem_bandwidth(mode="copy")
    gpu_mem_bandwidth(mode="triad")
