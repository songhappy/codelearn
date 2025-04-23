# Feature	register_full_backward_hook	register_hook (on param)
# Hook Level	Module (e.g., entire nn.Linear)	Individual Parameter (e.g., weight)
# Trigger Time	After backward pass on the module	During gradient computation
# Input Arguments	(module, grad_input, grad_output)	(grad)
# Use Case	Inspecting module-level gradient behavior	Monitoring/modifying param gradients
# Can be used on Parameters?	❌ No	✅ Yes

import torch
import torch.nn as nn
import torchvision.models as models
import time

torch_device = (
    torch.xpu if torch.xpu.is_available()
    else torch.cuda if torch.cuda.is_available()
    else torch.cpu
)

device = (
    'xpu' if torch.xpu.is_available()
    else 'cuda' if torch.cuda.is_available()
    else 'cpu'
)

_BYTES_IN_GIB = 1024 ** 3

def get_memo():

    peak_memory_active = (
        torch_device.memory_stats().get("active_bytes.all.peak", 0) / _BYTES_IN_GIB
    )
    peak_memory_alloc = torch_device.max_memory_allocated(device) / _BYTES_IN_GIB
    peak_memory_reserved = torch_device.max_memory_reserved(device) / _BYTES_IN_GIB
    return (time.time(), peak_memory_active, peak_memory_alloc, peak_memory_reserved)


            

# Initialize ResNet50
model = models.resnet50().to(device)
model.train()  # Set model to training mode



def forward_hook(module, input, output):
    current,  peak_memory_active, peak_memory_alloc, peak_memory_reserved = get_memo()
    print(f"\n[FORWARD HOOK] Mudule: {module.__class__.__name__}", current,  peak_memory_active, peak_memory_alloc, peak_memory_reserved)
                
for name, module in model.named_modules():
    module.register_forward_hook(forward_hook)

# 🔹 Hook function for gradients
def backward_hook(module, grad_input, grad_output):
    current,  peak_memory_active, peak_memory_alloc, peak_memory_reserved = get_memo() 
    print(f"\n[BACKWARD HOOK] Module: {module.__class__.__name__}", current,  peak_memory_active, peak_memory_alloc, peak_memory_reserved)
    return

# Register the hook on leaf modules
for name, module in model.named_modules():
    # if len(list(module.children())) == 0:  # Only hook leaf modules
    module.register_full_backward_hook(backward_hook)

def grad_hook(param_name):
    def hook(grad):
        torch.xpu.synchronize()
        peak_memory_reserved = torch_device.max_memory_reserved(self._device) / (1024**3)
        peak_memory_allocated = torch_device.max_memory_allocated(self._device) / (1024**3)
        print(f"\n[BACKWARD] Parameter: {param_name} memory reserved", peak_memory_reserved)
        print(f"\n[BACKWARD] Parameter: {param_name} memory allocated", peak_memory_allocated)
        return grad  # Clone to prevent in-place modification
    return hook


# 🔹 Register gradient hooks on model parameters (Fix)
for name, param in model.named_parameters():
    if param.requires_grad:  
        param.register_hook(grad_hook(name))
    else:
        print(f"Skipping hook for frozen param: {name}")
    


# Dummy data
input_data = torch.randn(32, 3, 224, 224, device=device)
target = torch.randint(0, 1000, (32,), device=device)

# Loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 🔹 Hook function for the forward pass
# 🔹 Hook function for the forward pass
def forward_hook(module, input, output):
    print(f"\n[FORWARD] Layer: {module.__class__.__name__}")
    print(f"  Input Data Type: {input[0].dtype}, Size: {input[0].shape}")

    if hasattr(module, "weight"):
        print(f"  Weight Data Type: {module.weight.dtype}, Size: {module.weight.shape}")

    print(f"  Output Data Type: {output.dtype}, Size: {output.shape}")

# 🔹 Hook function for gradients with module name
def grad_hook(param_name):
    def hook(grad):
        print(f"\n[BACKWARD] Parameter: {param_name}")
        print(f"  Gradient Data Type: {grad.dtype}, Size: {grad.shape}")
        return grad.clone()  # Clone to prevent in-place modification
    return hook

# 🔹 Register forward hooks on all layers
for name, layer in model.named_modules():
    layer.register_forward_hook(forward_hook)

# 🔹 Register gradient hooks on model parameters (Fix with Names)
for name, param in model.named_parameters():
    if param.requires_grad:
        param.register_hook(grad_hook(name))
        
# 🔹 Training Loop
for epoch in range(2):
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# export CUDA_LAUNCH_BLOCKING=1
# export PYTORCH_CUDA_ALLOC_CONF=verbose_debug
