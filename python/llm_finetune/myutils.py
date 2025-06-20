
import os
import gc
import torch
from torch.utils.data import DataLoader, Dataset
import subprocess
import json
import re

def get_device_bkd():
    if torch.xpu.is_available():
        torch_device = torch.xpu
        backend = "xpu:xccl"
        device_type = "xpu"
    elif torch.cuda.is_available():
        torch_device = torch.cuda
        backend = "nccl"
        device_type = "cuda"
    else:
        torch_device = torch.device("cpu")
        backend = "gloo"
        device_type = "cpu"
    return device_type, backend, torch_device

device_type, backend, torch_device = get_device_bkd()

def set_torch_cpp_log_level():
    rank = int(os.environ.get("RANK", -1))  # torchrun sets this automatically
    if rank == 0:
        os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    else:
        os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"


def print_lora_linear_structure(module, name=""):
    print(f"\n{name} (LoRALinear)")
    for param_name, param in module.named_parameters():
        print(f"  └─ {param_name}: shape={tuple(param.shape)}")

def print_all_parameters(model):
    print(f"{'Layer':<60} {'Shape':<30} {'# Params':<12} {'Trainable'}")
    print("-" * 115)
    total_params = 0
    total_trainable = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            total_trainable += num_params
        print(f"{name:<60} {str(tuple(param.shape)):<30} {num_params:<12} {param.requires_grad}")
    print("-" * 115)
    print(f"Total Parameters:          {total_params:,}")
    print(f"Total Trainable Parameters: {total_trainable:,}")
    print(f"Total Frozen Parameters:    {total_params - total_trainable:,}")

def clean_cache():
    """
    Clean up the XPU or CUDA cache to free up memory.
    """
    # try:
    #     if torch.xpu.is_available():
    #         torch.xpu.empty_cache()
    #         print("XPU cache cleared.")
    #     elif torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    #         print("CUDA cache cleared.")
    #     else:
    #         print("No XPU or CUDA device available to clear cache.")
    # except Exception as e:
    #     print(f"Failed to clear device cache: {e}")

    try:
        collected = gc.collect()
        print(f"GC collected {collected} unreachable objects.")
    except Exception as e:
        print(f"Failed to run garbage collection: {e}")



def get_gpu_memory_used_from_nvidia_smi(tag, device_id=0):
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader", "-i", str(device_id)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        mem_used_mb = float(result.stdout.strip())
        mem_used_gb = mem_used_mb / 1024
        return f"[{tag}] nvidia-smi memory used (device {device_id}): {mem_used_gb:.2f} GB"

    except Exception as e:
        return f"[{tag}] nvidia-smi error (device {device_id}): {e}"


def get_xpu_memory_used_from_xpu_smi(tag, device_id=0):
    try:
        result = subprocess.run(
            ["xpu-smi", "stats", "-d", str(device_id), "-j"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        stats = json.loads(result.stdout)
        tile_level = stats.get("tile_level", [])

        total_mem_mb = 0.0
        for tile in tile_level:
            for metric in tile.get("data_list", []):
                if metric["metrics_type"] == "XPUM_STATS_MEMORY_USED":
                    total_mem_mb += metric["value"]

        total_mem_gb = total_mem_mb / 1024
        return (f"[{tag}] xpu-smi memory used (device {device_id}): {total_mem_gb:.2f} GB")

    except Exception as e:
        return (f"[{tag}] xpu-smi error (device {device_id}): {e}")


def grad_hook(param_name):
    def hook(grad):
        grad = grad.clone()  # Force trigger

        # Optional: also print PyTorch-reported peak memory
        torch.xpu.synchronize()     
        peak_memory_reserved = torch_device.max_memory_reserved(torch_device.current_device()) / (1024**3)
        peak_memory_alloc = torch_device.max_memory_allocated(torch_device.current_device()) / (1024**3)
        print(f"\n[BACKWARD] Parameter: {param_name} | Reserved Memory: {peak_memory_reserved:.2f} GB, Allocated Memory: {peak_memory_alloc:.2f} GB")
        print(get_xpu_memory_used_from_xpu_smi(f"\n[BACKWARD] Parameter: {param_name}"))
        return grad  # Return original grad; clone if needed
    return hook

def forward_hook(module, input, output):
    peak_memory_reserved = torch_device.max_memory_reserved(torch_device.current_device()) / (1024**3)
    peak_memory_alloc = torch_device.max_memory_allocated(torch_device.current_device()) / (1024**3)
    print(f"\n[FORWARD] Layer: {module.__class__.__name__} | Reserved Memory: {peak_memory_reserved:.2f} GB, Allocated Memory: {peak_memory_alloc:.2f} GB")
    print(get_xpu_memory_used_from_xpu_smi(f"Forward: {module.__class__.__name__}"))

def backward_hook(name):
    def hook(module, grad_input, grad_output):
        # Optional: also print PyTorch-reported peak memory
        torch.xpu.synchronize()     
        peak_reserved = torch_device.max_memory_reserved(torch_device.current_device()) / (1024**3)
        peak_allocated = torch_device.max_memory_allocated(torch_device.current_device()) / (1024**3)
        print(f"\n[MODULE BACKWARD HOOK] {name}")
        print(f"  grad_input shapes: {[g.shape if g is not None else None for g in grad_input]}")
        print(f"  grad_output shapes: {[g.shape if g is not None else None for g in grad_output]}")
        print(f"  Reserved: {peak_reserved:.2f} GB | Allocated: {peak_allocated:.2f} GB")
        print(get_xpu_memory_used_from_xpu_smi(f"Backward: {module.__class__.__name__}"))

    return hook

