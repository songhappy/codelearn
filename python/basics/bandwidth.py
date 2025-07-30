import torch
import time

# Parameters
MB = 1024 * 1024
size_mb = 512
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

print(f"Estimated sustained memory bandwidth: {bandwidth_gbps:.2f} GB/s")
