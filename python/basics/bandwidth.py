import torch
import time

# Large memory size: 512 MB (float32 = 4B)
MB = 1024 * 1024
num_elems = 512 * MB // 4  # 512MB in float32
a = torch.rand(num_elems, device="xpu")
b = torch.empty_like(a)

# Warm-up
for _ in range(5):
    b.copy_(a)
torch.xpu.synchronize()

# Time repeated memory copies
N = 1000
start = time.perf_counter()
for _ in range(N):
    b.copy_(a)
torch.xpu.synchronize()
end = time.perf_counter()

# Each copy = read + write = 2× traffic
bytes_transferred = 2 * a.numel() * 4 * N
bandwidth_gbps = bytes_transferred / (end - start) / 1e9

print(f"Estimated sustained memory bandwidth: {bandwidth_gbps:.2f} GB/s")
