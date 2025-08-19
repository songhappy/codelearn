import torch
import time

# Parameters
def get_bandwidth(size_mb):
    """Estimate the sustained memory bandwidth of the XPU.
    This function allocates a large array, performs a copy operation,
    and measures the time taken to estimate bandwidth.
    """
    MB = 1024 * 1024
    num_elems = size_mb * MB // 4  # float32 = 4B

    # Allocate
    a = torch.rand(num_elems, device="xpu")
    b = torch.empty_like(a)

    # Confirm dtype
    assert a.dtype == torch.float32

    # Warm-up
    for _ in range(5):
        b.copy_(a)
    torch.xpu.synchronize()

    # Benchmark
    N = 1000
    start = time.perf_counter()
    for _ in range(N):
        b.copy_(a)
    torch.xpu.synchronize()
    end = time.perf_counter()

    # Compute bandwidth
    bytes_transferred = 2 * a.numel() * 4 * N  # read + write
    bandwidth_gbps = bytes_transferred / (end - start) / 1e9
    return bandwidth_gbps

def get_bandwidth_streams(size_mb):
    MB = 1024 * 1024
    num_elems = size_mb * MB // 4  # float32 = 4B

    # Allocate multiple buffers
    a_list = [torch.rand(num_elems, device="xpu") for _ in range(4)]
    b_list = [torch.empty_like(a) for a in a_list]

    # Warm up
    for a, b in zip(a_list, b_list):
        b.copy_(a)
    torch.xpu.synchronize()

    # Time many concurrent copies using streams
    streams = [torch.xpu.Stream() for _ in range(4)]
    N = 500
    start = time.perf_counter()

    for _ in range(N):
        for i in range(4):
            with torch.xpu.stream(streams[i]):
                b_list[i].copy_(a_list[i])
    torch.xpu.synchronize()

    end = time.perf_counter()

    # Total bytes = read + write × num tensors × N
    total_bytes = 4 * 2 * num_elems * 4 * N
    bw = total_bytes / (end - start) / 1e9

    print(f"Pushed sustained memory bandwidth: {bw:.2f} GB/s")
    return get_bandwidth(size_mb)


if __name__ == "__main__":

    size_mb =[512, 1024, 2048, 4096, 8192, 16384]  # Sizes in MB

    for size in size_mb:
        bandwidth_gbps = get_bandwidth(size)
        print(f"Estimated sustained memory bandwidth: {bandwidth_gbps:.2f} GB/s")
    for size in size_mb:
        bandwidth_gbps = get_bandwidth_streams(size)
        print(f"Estimated sustained memory bandwidth (streams): {bandwidth_gbps:.2f} GB/s")