import time
import torch

# --------------------------
# Backend/Device Abstraction
# --------------------------
def pick_backend():
    has_xpu  = hasattr(torch, "xpu") and torch.xpu.is_available()
    has_cuda = torch.cuda.is_available()
    if has_xpu:
        return "xpu"
    if has_cuda:
        return "cuda"
    return "cpu"

BACKEND = pick_backend()

def device():
    if BACKEND == "xpu":
        return torch.device("xpu")
    if BACKEND == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")

def synchronize():
    if BACKEND == "xpu":
        torch.xpu.synchronize()
    elif BACKEND == "cuda":
        torch.cuda.synchronize()
    else:
        torch.cpu.synchronize() if hasattr(torch, "cpu") and hasattr(torch.cpu, "synchronize") else None

def Stream():
    if BACKEND == "xpu":
        return torch.xpu.Stream()
    if BACKEND == "cuda":
        return torch.cuda.Stream()
    # CPU has no streams; return a dummy object/context
    class _NoopStream:
        pass
    return _NoopStream()

def stream_ctx(stream):
    if BACKEND == "xpu":
        return torch.xpu.stream(stream)
    if BACKEND == "cuda":
        return torch.cuda.stream(stream)
    # CPU: no-op context
    from contextlib import contextmanager
    @contextmanager
    def _noop():
        yield
    return _noop()

# --------------------------
# Benchmarks
# --------------------------
@torch.no_grad()
def get_bandwidth(size_mb: int, iters: int = 1000) -> float:
    """
    Estimate sustained device memory bandwidth (GB/s) via repeated device->device copies.
    Works on XPU, CUDA, and CPU (for correctness).
    """
    MB = 1024 * 1024
    elem_bytes = 4  # float32
    num_elems = size_mb * MB // elem_bytes

    # Allocate on the selected device
    dev = device()
    a = torch.rand(num_elems, device=dev, dtype=torch.float32)
    b = torch.empty_like(a)

    assert a.dtype == torch.float32

    # Warm-up
    for _ in range(5):
        b.copy_(a)
    synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        b.copy_(a)
    synchronize()
    end = time.perf_counter()

    # read + write per copy
    bytes_per_copy = 2 * a.numel() * a.element_size()
    total_bytes = bytes_per_copy * iters
    gbps = total_bytes / (end - start) / 1e9
    return gbps


@torch.no_grad()
def get_bandwidth_streams(size_mb: int, num_streams: int = 4, iters: int = 500) -> float:
    """
    Estimate sustained bandwidth using multiple concurrent streams.
    On CPU (no streams), it behaves as a simple loop (no concurrency).
    """
    MB = 1024 * 1024
    elem_bytes = 4  # float32
    num_elems = size_mb * MB // elem_bytes

    dev = device()
    a_list = [torch.rand(num_elems, device=dev, dtype=torch.float32) for _ in range(num_streams)]
    b_list = [torch.empty_like(a) for a in a_list]

    # Warm-up
    for a, b in zip(a_list, b_list):
        b.copy_(a)
    synchronize()

    # Prepare streams (no-op on CPU)
    streams = [Stream() for _ in range(num_streams)]

    start = time.perf_counter()
    for _ in range(iters):
        for i in range(num_streams):
            with stream_ctx(streams[i]):
                b_list[i].copy_(a_list[i])
    synchronize()
    end = time.perf_counter()

    # total bytes = (#tensors) * (read+write per copy) * iters
    bytes_per_copy = 2 * num_elems * elem_bytes
    total_bytes = num_streams * bytes_per_copy * iters
    gbps = total_bytes / (end - start) / 1e9

    print(f"[{BACKEND}] Streams ({num_streams}) sustained bandwidth: {gbps:.2f} GB/s")
    # Also return single-stream estimate for comparison consistency (optional)
    return get_bandwidth(size_mb)


if __name__ == "__main__":
    print(f"Selected backend: {BACKEND}, device: {device()}")

    # Sizes in MB (adjust if you hit OOM on your device)
    sizes_mb = [512, 1024, 2048, 4096, 8192, 16384]

    for size in sizes_mb:
        try:
            bw = get_bandwidth(size)
            print(f"[{BACKEND}] Size {size:>6} MB | Sustained bandwidth: {bw:.2f} GB/s")
        except RuntimeError as e:
            print(f"[{BACKEND}] Size {size} MB | Skipped due to error: {e}")

    for size in sizes_mb:
        try:
            bw_streams = get_bandwidth_streams(size)
            print(f"[{BACKEND}] Size {size:>6} MB | Sustained bandwidth (streams): {bw_streams:.2f} GB/s")
        except RuntimeError as e:
            print(f"[{BACKEND}] Size {size} MB | Streams skipped due to error: {e}")
