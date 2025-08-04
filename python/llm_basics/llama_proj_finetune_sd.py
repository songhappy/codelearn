import numpy as np

b_cap = 0.85
Prompt = 512  # sequence length

# model_config for LLaMA 3.1 8B
Hidden = 4096
AttHeads = 32
Headdim = Hidden / AttHeads
KVHidden = Headdim * 2  # typical KV projection size
Interm = 14336
Vocab = 128256
Block = 32

# hw_config (e.g., Intel PVC BF16)
B_eff = 0.9
B = 1050 * 1e9 * B_eff  # Bandwidth in bytes/sec
T_eff = 1.0
T = 419 * 1e12 * T_eff  # Compute throughput in FLOPs/sec (BF16)

kv_dtype = 2
w_dtype = 2      # weight dtype (BF16 or FP16 = 2 bytes)
act_dtype = 2    # activation dtype

def proj_llama_finetune(N):
    # --- Model parameter sizes ---
    C = Hidden * (3 * Interm + Hidden * 2 + KVHidden * 2)  # per-block param count
    block_weight = C * w_dtype * Block  # bytes
    embed_weight = Vocab * Hidden * w_dtype  # shared (tied) weights
    total_weight_sz = block_weight + embed_weight

    # --- Compute cost (fwd + bwd) ---
    ops_per_token = 2 * C * 3  # 3x for fwd+bwd
    total_ops = N * Prompt * ops_per_token * Block
    fbw_time = total_ops / T

    # --- Optimizer step ---
    optimizer_ops = C * Block * 2  # Adam-like update
    time_optimizer = optimizer_ops / T

    # --- Total model time ---
    time_model = fbw_time + time_optimizer

    # --- Memory usage ---
    act_mem = N * Prompt * Hidden * act_dtype * Block  # activation memory
    param_size = C * Block * w_dtype
    grad_size = C * Block * w_dtype
    opt_states = param_size * 2  # Adam: m + v

    model_mem = param_size + grad_size + opt_states
    total_mem = act_mem + model_mem

    # --- Bandwidth-bound time ---
    time_memory = total_mem / B
    total_time = max(time_model, time_memory)

    # --- Final stats ---
    mem_gib = total_mem / (1024 ** 3)
    model_gib = total_weight_sz / (1024 ** 3)
    tok_per_sec = (N * Prompt) / total_time

    return mem_gib, model_gib, total_time, tok_per_sec

# === Sweep ===
batch_sizes, memo_sizes, model_sizes = [], [], []
pre_times, token_per_secs = [], []

for N in [1, 2, 4]:
    memo_gb, model_gb, time, tps = proj_llama_finetune(N)
    batch_sizes.append(N)
    memo_sizes.append(memo_gb)
    model_sizes.append(model_gb)
    pre_times.append(time)
    token_per_secs.append(tps)

# === Output ===
print("batch size, ISL/OSL, Total time (s), T/S, Model size (GiB), Total memory (GiB)")
for a, b, c, d, e, f in zip(batch_sizes, [Prompt]*len(batch_sizes), pre_times, token_per_secs, model_sizes, memo_sizes):
    print(f"{a},{b},{c:.4f},{d:.2f},{e:.2f},{f:.2f}")
