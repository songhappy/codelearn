import numpy as np

# === Model Configuration ===
Hidden = 4096                     # Model hidden size
AttHeads = 32                    # Number of attention heads
Headdim = Hidden / AttHeads      # Head dimension
KVHidden = Headdim * 2           # Key/Value projection dimension (typical)
Interm = 14336                   # FFN intermediate size
Vocab = 128256                   # Vocabulary size
Block = 32                       # Number of transformer blocks

Prompt = 512                     # Sequence length per sample
w_dtype = 2                      # Weight precision (bytes) → BF16 = 2
act_dtype = 2                    # Activation precision (BF16)

# === Hardware Configuration ===
B_eff = 0.9
B_base = 1050e9                  # Theoretical memory bandwidth (bytes/sec)
B = B_base * B_eff               # Effective bandwidth

T_eff = 1.0
T_base = 419e12                  # Theoretical compute throughput (FLOPs/sec)
T = T_base * T_eff               # Effective compute throughput

comm_bw = 100e9                  # Inter-device bandwidth (bytes/sec)
num_devices = 4                  # FSDP devices (single node)
scaling_eff = 0.95               # Efficiency of distributed training

def proj_llama_finetune_fsdp2(N):
    """
    Estimate memory, time, and throughput for fine-tuning LLaMA 8B using FSDP2
    across 4 devices on a single node.
    
    Args:
        N (int): global batch size (distributed across devices)
    Returns:
        tuple of memory per device (GiB), model size (GiB),
        total step time (s), and token throughput (tokens/sec)
    """
    per_device_N = N // num_devices  # batch size per device

    # === Model Size and FLOPs ===
    C = Hidden * (3 * Interm + Hidden * 2 + KVHidden * 2)  # per-layer param count

    # --- Compute time: forward + backward + optimizer ---
    ops_fwd_bwd = N * Prompt * C * 3 * 2 * Block  # 3x ops per token (GEMMs, etc.)
    time_compute = ops_fwd_bwd / (T * num_devices)

    optimizer_ops = C * Block * 2  # Adam: 2 ops per param
    time_optimizer = optimizer_ops / (T * num_devices)

    # --- Communication: AllReduce of gradients ---
    
    grad_size = C * Block * w_dtype  # total gradient size (bytes)
    comm_bytes = 2 * grad_size * np.log2(num_devices)  
    time_comm = comm_bytes / comm_bw  # communication time

    # === Memory Per Device ===

    # --- Activation memory (not sharded) ---
    act_mem = per_device_N * Prompt * Hidden * act_dtype * Block

    # --- FSDP2 sharded model memory ---
    param_shard = C * Block * w_dtype / num_devices
    grad_shard = grad_size / num_devices
    opt_state = param_shard * 2  # Adam: m + v
    model_mem = param_shard + grad_shard + opt_state

    total_mem = act_mem + model_mem  # per-device memory footprint
    time_memory = total_mem / B      # bandwidth-bound step time

    # === Roofline Time: compute or memory bound + communication ===
    total_time = max(time_compute + time_optimizer, time_memory) + time_comm

    # === Token Throughput ===
    tokens_per_sec = (N * Prompt) / total_time * scaling_eff

    # === Memory Stats ===
    mem_gib = total_mem / (1024 ** 3)
    model_total = param_shard * num_devices + embed_weight()
    model_gib = model_total / (1024 ** 3)

    return mem_gib, model_gib, total_time, tokens_per_sec

def embed_weight():
    """Embedding layer (tied input/output projection)"""
    return Vocab * Hidden * w_dtype

# === Sweep Over Global Batch Sizes ===
batch_sizes = [8, 16, 32, 64, 128, 256]
results = []

for N in batch_sizes:
    mem_gb, model_gb, time, tps = proj_llama_finetune_fsdp2(N)
    results.append((N, Prompt, time, tps, model_gb, mem_gb))

# === Output ===
print("global_batch, ISL/OSL, Total time (s), T/S, Model size (GiB), Memory/device (GiB)")
for r in results:
    print(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f}")
