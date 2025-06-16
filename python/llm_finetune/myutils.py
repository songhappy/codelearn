
import os
import gc
import torch
from torch.utils.data import DataLoader, Dataset
from torchtune.utils import get_torch_device_namespace
import subprocess
import json

torch_device = get_torch_device_namespace()
from torchtune import  utils

device = (
    'xpu' if torch.xpu.is_available()
    else 'cuda' if torch.cuda.is_available()
    else 'cpu'
)
def set_torch_cpp_log_level():
    rank = int(os.environ.get("RANK", -1))  # torchrun sets this automatically
    if rank == 0:
        os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    else:
        os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
def get_memo():

    peak_memory_active = (
        torch_device.memory_stats().get("active_bytes.all.peak", 0) / 1024 ** 3
    )
    peak_memory_alloc = torch_device.max_memory_allocated(device) /  1024 ** 3
    peak_memory_reserved = torch_device.max_memory_reserved(device) /  1024 ** 3
    print(peak_memory_active, peak_memory_alloc, peak_memory_reserved)
    return

log = utils.get_logger("DEBUG")

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
    try:
        if torch.xpu.is_available():
            torch.xpu.empty_cache()
            print("XPU cache cleared.")
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")
        else:
            print("No XPU or CUDA device available to clear cache.")
    except Exception as e:
        print(f"Failed to clear device cache: {e}")

    try:
        collected = gc.collect()
        print(f"GC collected {collected} unreachable objects.")
    except Exception as e:
        print(f"Failed to run garbage collection: {e}")



def print_xpu_memory_used_from_xpu_smi(tag, device_id=0):
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
        print(f"[{tag}] xpu-smi memory used (device {device_id}): {total_mem_gb:.2f} GB")

    except Exception as e:
        print(f"[{tag}] xpu-smi error (device {device_id}): {e}")


def grad_hook(param_name):
    def hook(grad):
        torch.xpu.synchronize()

        # Optional: also print PyTorch-reported peak memory
        torch.xpu.synchronize()     
        peak_memory_reserved = torch_device.max_memory_reserved(torch_device.current_device()) / (1024**3)
        peak_memory_alloc = torch_device.max_memory_allocated(torch_device.current_device()) / (1024**3)
        print(f"\n[BACKWARD] Parameter: {param_name} | Reserved Memory: {peak_memory_reserved:.2f} GB, Allocated Memory: {peak_memory_alloc:.2f} GB")
        print_xpu_memory_used_from_xpu_smi(f"\n[BACKWARD] Parameter: {param_name}")
        return grad  # Return original grad; clone if needed
    return hook

def forward_hook(module, input, output):
    peak_memory_reserved = torch_device.max_memory_reserved(torch_device.current_device()) / (1024**3)
    peak_memory_alloc = torch_device.max_memory_allocated(torch_device.current_device()) / (1024**3)
    print(f"\n[FORWARD] Layer: {module.__class__.__name__} | Reserved Memory: {peak_memory_reserved:.2f} GB, Allocated Memory: {peak_memory_alloc:.2f} GB")
    print_xpu_memory_used_from_xpu_smi(f"Forward: {module.__class__.__name__}")
