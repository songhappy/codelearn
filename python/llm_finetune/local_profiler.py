import torch
import torchvision.models as models
import torch.profiler

# Check for A100 GPU


if torch.xpu.is_available():
    device = torch.device("xpu:0")
    torch_device = torch.xpu
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch_device = torch.cuda
else:
    device = torch.device("cpu")
    torch_device = torch


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
        torch.profiler.ProfilerActivity.CUDA,
        torch.profiler.ProfilerActivity.CPU,
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
print(prof.key_averages().table(sort_by="xpu_time_total", row_limit=10))
# print(prof.key_averages().table(sort_by="xpu_memory_usage", max_name_column_width=30, row_limit=10))

        

# print(prof.key_averages().table(sort_by="xpu_time_total", row_limit=10))
