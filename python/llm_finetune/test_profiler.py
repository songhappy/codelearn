import torch
import torchvision.models as models
import torch.profiler

# Check for A100 GPU
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
torch.xpu.max_memory_allocated
torch.cuda.memory_stats

torch_device = torch.xpu
# torch_device = torch.cuda

peak_memory_active = torch_device.memory_stats().get("active_bytes.all.peak", 0) / (
    1024**3
)
peak_mem_alloc = torch_device.max_memory_allocated(device) / (1024**3)
peak_mem_reserved = torch_device.max_memory_reserved(device) / (1024**3)
torch_device.reset_peak_memory_stats(device)

# Initialize ResNet50
model = models.resnet50().to(device)
model.train()  # Set model to training mode

# Dummy data for profiling
input_data = torch.randn(32, 3, 224, 224).to(device)  # Batch size 32, ImageNet size
target = torch.randint(0, 1000, (32,)).to(device)  # Dummy labels for classification

# Loss function
criterion = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

prof = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.XPU,
    ],
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True
)

# Start the profiler (but only record one iteration)
prof.start()

# Profiling
for idx in range(20):  # Simulate training loop
    optimizer.zero_grad()

    # Forward pass
    output = model(input_data)
    loss = criterion(output, target)

    # Backward pass
    loss.backward()
    optimizer.step()

    torch.xpu.synchronize()  # Ensure proper synchronization

    # Profile only the 7th iteration (idx == 6)
    if idx == 6:
        prof.step()  # Record only this iteration

# Stop the profiler
prof.stop()

# Print profiling results
print(prof.key_averages().table(sort_by="xpu_time_total", row_limit=100))
print(prof.key_averages().table(sort_by="xpu_memory_usage", max_name_column_width=30, row_limit=100))

        

# print(prof.key_averages().table(sort_by="xpu_time_total", row_limit=10))
