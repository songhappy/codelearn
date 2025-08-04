
import numpy as np
import utils

N = 8                   # Batch size
Prompt = 1024          # Prompt token length
Gen = 1024             # Number of generated tokens
Hidden = 4096          # Hidden size (model dimension)
AttHeads = 32          # Number of attention heads
Headdim = Hidden / AttHeads  # Dimension per attention head
KVHidden = Headdim * 8       # Key/Value projection dim
Interm = 14336         # Intermediate size in FFN
Vocab = 128256         # Vocabulary size
Block = 32             # Number of Transformer blocks
Causal = True          # Whether attention is causal (true for generation)

B_eff = 0.9
B = 920e9 * B_eff      # Bandwidth in bytes/sec (e.g., HBM)
T_eff = 1
T = 1066e12 * T_eff    # Compute throughput in FLOPs/sec

kv_dtype = 1           # Assume KV cache in 1 byte precision
w_dtype = 0.5          # Model weights in FP8 or similar
q_group = 128          # Quantization group size
fw_dtype = 2           # Forward pass output (e.g., logits) in BF16 
L3_size = 203 * 1024 * 1024  # L3 cache size (bytes)


def proj_llama():
    C = Hidden * (3 * Interm + Hidden * 2 + KVHidden * 2)
    # last gemm
    logitw_size = Hidden * Vocab

    prompt_kv = 2 * N * Prompt * KVHidden * kv_dtype
    time_rope = prompt_kv / B
    time_rsmnorm = 3 * prompt_kv / B

    sdpa = 2 * N * Prompt * Prompt * Headdim * AttHeads
    if not Causal:
        sdpa *= 2

    time_sdpa = sdpa / T

    # prefill roughly equal to promp compute time in each block
    time_prefill_1block = 2 * Prompt * N * C / T + time_rope + time_sdpa + time_rsmnorm
    time_prefill_model = time_prefill_1block * Block

    # Generation in 1 transformer block, kv-history and model
    # weights read or compute when batch size N is large
    # The max() is an over simplification which could turn into
    # other sophisticated functions
    size_kv = [2 * N * (Prompt + i) * KVHidden * kv_dtype for i in range(Gen)]
    t_size_kv = 2 * N * (Prompt + Gen) * KVHidden * kv_dtype * Block
    weight_sz = C * w_dtype
    gen_ops = 2 * N * C
    
    B_compute = B * b_cap
    time_gen_1block = [each_kv / B_compute + max(weight_sz/B_compute, gen_ops/T) for each_kv in size_kv]

    # Last gemm for the logits, non-trivial if large vocabulary
    time_next = max(logitw_size*fw_dtype/B, 2*N*logitw_size/(T/(fw_dtype/w_dtype)))
    time_gen_token = [ gen_1block * Block + time_next for gen_1block in time_gen_1block ]

    # total time prefill, logit gemm and generation of each token
    time = time_prefill_model + time_next + np.sum(time_gen_token)
    model_sz = weight_sz * Block + logitw_size * fw_dtype * 2
    total_sz = model_sz + t_size_kv

    # in GiB
    kv_cache_size = t_size_kv/1024/1024/1024
    memo_size = total_sz/1024/1024/1024
    w_size = model_sz / 1024/1024/1024
    token_per_sec = N * Gen / time
    t_per_sec_vllm =  N * (Gen + Prompt)/time
    pre_time = time_prefill_model + time_next
    next_min = time_gen_token[0]
    next_max =  time_gen_token[-1]
    p90_latency, p99_lentency = utils.quntiles_90_99(time_gen_token)
    genertion_time = np.sum(time_gen_token)
    gen_prefill_ratio = np.sum(time_gen_token) / (time_prefill_model + time_next)
    Compute_Memory_ratio= (gen_ops/T)/(weight_sz/B)
    
    return kv_cache_size, memo_size, pre_time, p90_latency, token_per_sec

batch_sizes = []
kv_cache_sizes = []
memo_sizes = []
pre_times = []
p90_latencys = []
token_per_secs = []
iosls = []

threshold = 50  # in ms
for N in [1, 2, 512, 1024]:
    (kv_cache_size, memo_size, pre_time, p90_latency, token_per_sec) = proj_llama()
    batch_sizes.append(N)
    iosls.append((Prompt,Gen))
    kv_cache_sizes.append(kv_cache_size)
    memo_sizes.append(memo_size)
    pre_times.append(pre_time)
    p90_latencys.append(p90_latency)
    token_per_secs.append(token_per_sec)
    if p90_latency*1000 > threshold:
        break
print("batch size, ISL/OSL, Prefill time, Next_p90_latency, T/S, KV-cache size in GiB, Memory size in GiB")
for a, (isl, osl), c, d, e, f, g in zip(batch_sizes, iosls, pre_times, p90_latencys, token_per_secs, kv_cache_sizes, memo_sizes):
    print(f"{a},{isl}/{osl},{c},{d},{e},{f},{g}")