import torch
import torch.nn as nn

print("✅ Script started")

# Device setup
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
torch_device = torch.xpu if device.type == "xpu" else torch.cuda

# Dummy memory print
def print_xpu_memory_used_from_xpu_smi(tag=""):
    print(f"{tag} [Simulated XPU SMI]")

# Define a simple model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)  # Frozen
        self.linear2 = nn.Linear(20, 1)   # Trainable

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        return self.linear2(x)

model = MyModel().to(device)

# Freeze linear1
for param in model.linear1.parameters():
    param.requires_grad = False

# -------------------------------
# ✅ Module-level FORWARD hook
# -------------------------------
def forward_module_hook(module, input, output):
    peak_reserved = torch_device.max_memory_reserved(torch_device.current_device()) / (1024**3)
    peak_allocated = torch_device.max_memory_allocated(torch_device.current_device()) / (1024**3)
    print(f"\n[MODULE FORWARD HOOK] {module.__class__.__name__}")
    print(f"  Reserved: {peak_reserved:.2f} GB | Allocated: {peak_allocated:.2f} GB")
    print_xpu_memory_used_from_xpu_smi(f"[MODULE FORWARD] {module.__class__.__name__}")

# -------------------------------
# ✅ Module-level BACKWARD hook
# -------------------------------
def backward_module_hook(name):
    def hook(module, grad_input, grad_output):
        torch.xpu.synchronize()
        peak_reserved = torch_device.max_memory_reserved(torch_device.current_device()) / (1024**3)
        peak_allocated = torch_device.max_memory_allocated(torch_device.current_device()) / (1024**3)
        print(f"\n[MODULE BACKWARD HOOK] {name}")
        print(f"  grad_input shapes: {[g.shape if g is not None else None for g in grad_input]}")
        print(f"  grad_output shapes: {[g.shape if g is not None else None for g in grad_output]}")
        print(f"  Reserved: {peak_reserved:.2f} GB | Allocated: {peak_allocated:.2f} GB")
    return hook

# -------------------------------
# ✅ Parameter-level BACKWARD hook
# -------------------------------
def backward_param_hook(param_name):
    def hook(grad):
        torch.xpu.synchronize()
        peak_reserved = torch_device.max_memory_reserved(torch_device.current_device()) / (1024**3)
        peak_allocated = torch_device.max_memory_allocated(torch_device.current_device()) / (1024**3)
        print(f"\n[PARAMETER BACKWARD HOOK] {param_name}")
        print(f"  grad shape: {grad.shape} | norm: {grad.norm():.4f}")
        print(f"  Reserved: {peak_reserved:.2f} GB | Allocated: {peak_allocated:.2f} GB")
        return grad
    return hook

# -------------------------------
# 🔧 Register all hooks
# -------------------------------

# Module hooks
for name, module in model.named_modules():
    module.register_forward_hook(forward_module_hook)
    if any(p.requires_grad for p in module.parameters()):
        module.register_full_backward_hook(backward_module_hook(name))
        print(f"✅ Registered module forward + backward hook on: {name}")

# Parameter hooks
for name, param in model.named_parameters():
    if param.requires_grad:
        param.register_hook(backward_param_hook(name))
        print(f"✅ Registered parameter backward hook on: {name}")
    else:
        print(f"🚫 Skipped frozen param: {name}")

# -------------------------------
# 🔁 Run dummy training step
# -------------------------------
x = torch.randn(4, 10, device=device)
target = torch.randn(4, 1, device=device)
criterion = nn.MSELoss()

output = model(x)
loss = criterion(output, target)

print("\n--- Backward pass ---")
loss.backward()
