# llama_project_train_mm.py
# this code it to project throughput and memory usage of training a llama model
# ========================= Transformer Math + Roofline + Adam =========================
# - HuggingFace config names (hidden_size, num_hidden_layers, ...)
# - Math aliases with equations in docstrings
# - Parameter counts (no bias)
# - FLOPs/token, per-step (forward; training with checkpoint factor)
# - Adam/AdamW FLOPs & bytes per step
# - Bytes model (I/O lower bound for GEMMs + AI for attention)
# - Roofline timing (compute vs memory) and throughput
# - Memory footprint (persistent + rough activation bounds)
# =====================================================================================

from dataclasses import dataclass
from typing import Dict, Tuple
from python.llama.llamaspec import LlamaSpec

# ---------- pretty helpers ----------
def human_bytes(x: float) -> str:
    units = ["B","KB","MB","GB","TB","PB"]; i = 0; x = float(x)
    while abs(x) >= 1024 and i < len(units)-1: x /= 1024.0; i += 1
    return f"{x:.2f} {units[i]}"

def human_num(x: float) -> str:
    x = float(x)
    for u in ["","K","M","B","T","P"]:
        if abs(x) < 1000: return f"{x:.2f}{u}"
        x /= 1000.0
    return f"{x:.2f}E"

# ---------- core math helpers ----------
def gemm_flops(M: int, K: int, N: int) -> int:
    r"""FLOPs for A(M×K) @ B(K×N) -> C(M×N) with MAC=2 FLOPs:  2·M·K·N"""
    return 2 * M * K * N

def gemm_bytes_step(M: int, K: int, N: int, b: int) -> int:
    r"""I/O lower bound bytes/step: read A + read B + write C ≈ 2·(MK + KN + MN)·b"""
    return 2 * b * (M*K + K*N + M*N)

def gemm_bytes_token(M: int, K: int, N: int, b: int) -> float:
    r"""Per-token bytes = [2·(MK + KN + MN)·b] / M"""
    return gemm_bytes_step(M, K, N, b) / max(M, 1)


# ---------- dtype bytes ----------
@dataclass(frozen=True)
class DTypeBytes:
    w: int = 2    # weights dtype (bf16=2)
    act: int = 2  # activations
    grad: int = 2
    m: int = 2    # Adam moments in bf16 (use 4 for fp32)
    v: int = 2

# ---------- training & hardware ----------
@dataclass(frozen=True)
class TrainSpec:
    batch_size: int      # B
    seq_len: int         # S
    grad_accum: int = 1  # G
    checkpoint: bool = True

    # aligned names / toggles
    use_flash_attn: bool = True
    include_sdpa_bytes: bool = False
    include_act_io_linear: bool = True
    act_coeff_per_layer: float = 2.0

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.seq_len * self.grad_accum

@dataclass(frozen=True)
class HardwareSpec:
    bandwidth_Bps_raw: float   # raw HBM bytes/s
    bandwidth_eff: float       # efficiency (0..1)
    compute_Fps_raw: float     # raw FLOPs/s
    compute_eff: float         # efficiency (0..1)

    @property
    def B(self) -> float:      # effective memory bandwidth
        return self.bandwidth_Bps_raw * self.bandwidth_eff

    @property
    def T(self) -> float:      # effective compute throughput
        return self.compute_Fps_raw * self.compute_eff

# ---------- bytes model (forward token; training step) ----------
def bytes_forward_per_token(spec: LlamaSpec, train: TrainSpec, dtypes: DTypeBytes) -> float:
    """
    Non-attention GEMMs use IO lower bound:
      bytes ≈ 2·(MK + KN + MN)·b  ; per token -> divide by M_micro = B·S.
    Attention bytes via arithmetic intensity AI (optional):
      FLOPs/token(attn) = L·4·D·S ; bytes(attn) ≈ FLOPs / AI
        • with FlashAttention: AI ≈ S/2 (bf16)
        • without FA: AI ≈ 82 + 0.005·S  (conservative)
    LM head uses same GEMM bound.
    """
    D, Dkv, Dff, L, V = spec.hidden_size, spec.kv_dim_total, spec.intermediate_size, spec.num_hidden_layers, spec.vocab_size
    S, b = train.seq_len, dtypes.act
    M_micro = train.batch_size * train.seq_len

    # per-layer non-attn (per token)
    q  = gemm_bytes_token(M_micro, D,    D,    b)
    k  = gemm_bytes_token(M_micro, D,    Dkv,  b)
    v  = gemm_bytes_token(M_micro, D,    Dkv,  b)
    o  = gemm_bytes_token(M_micro, D,    D,    b)
    w1 = gemm_bytes_token(M_micro, D,    Dff,  b)
    w3 = gemm_bytes_token(M_micro, D,    Dff,  b)
    w2 = gemm_bytes_token(M_micro, Dff,  D,    b)
    nonattn = L * (q+k+v+o+w1+w3+w2)

    # attention bytes via AI (optional to match other script)
    attn_bytes = 0.0
    if train.include_sdpa_bytes:
        F_attn_token = L * (4 * D * S)
        AI = (S/2.0) if train.use_flash_attn else (82.0 + 0.005*S)
        attn_bytes = F_attn_token / max(AI, 1e-9)

    # LM head bytes
    lm = gemm_bytes_token(M_micro, D, V, b)
    return nonattn + attn_bytes + lm

def bytes_train_per_step(spec: LlamaSpec, train: TrainSpec, dtypes: DTypeBytes) -> float:
    """
    FW+BW bytes/step (weight I/O model, aligned with your other script):
      bytes_fwbw = 2·P·w·G + P·grad + (acts) + (optional SDPA bytes)
    """
    P = spec.total_params
    G = train.grad_accum
    Bsz, S = train.batch_size, train.seq_len
    L, D = spec.num_hidden_layers, spec.hidden_size
    M = Bsz * S * G

    # weights read+write per FW+BW pass, repeated grad_accum times
    bytes_weights_fw_bw = 2.0 * P * dtypes.w * G

    # gradients written once per step
    bytes_grad = P * dtypes.grad

    # coarse activations
    bytes_act = 0.0
    if train.include_act_io_linear:
        bytes_act = M * L * (train.act_coeff_per_layer * D * dtypes.act)

    # optional SDPA bytes via AI model (usually disabled)
    bytes_sdpa = 0.0
    if train.include_sdpa_bytes:
        alpha = 4.0 if train.checkpoint else 3.0
        F_sdpa_train = (alpha * L * 4.0 * D * S) * M
        AI = (S/2.0) if train.use_flash_attn else (82.0 + 0.005*S)
        bytes_sdpa = F_sdpa_train / max(AI, 1e-9)

    return bytes_weights_fw_bw + bytes_grad + bytes_act + bytes_sdpa

def adam_bytes_per_step(spec: LlamaSpec, dtypes: DTypeBytes) -> int:
    """
    Optimizer bytes per step (generic formula using dtypes):
      read:  W + g + m + v
      write: W + m + v
      total: (2·W + g + 2·m + 2·v) bytes per param
      (bf16 everywhere ⇒ ~14 B/param; fp32 moments ⇒ higher)
    """
    P = spec.total_params
    return P * ( (dtypes.w + dtypes.grad + dtypes.m + dtypes.v) + (dtypes.w + dtypes.m + dtypes.v) )

def _which_bottleneck(t_compute: float, t_memory: float, name: str, eps: float = 1e-9):
    """
    Decide which bound dominates and return a label + an explanatory string.
    ratio = t_compute / t_memory (>1 ⇒ compute slower; <1 ⇒ memory slower).
    """
    ratio = (t_compute + eps) / (t_memory + eps)
    if abs(ratio - 1.0) <= 0.02:   # within ~2% → treat as 'tied'
        label = "tied"
    elif ratio > 1.0:
        label = "compute"
    else:
        label = "memory"
    msg = (f"{name} bottleneck: {label} "
           f"(compute={t_compute:.4f}s, memory={t_memory:.4f}s, ratio={ratio:.2f})")
    return label, ratio, msg

# ---------- roofline timing ----------
def roofline_step(spec: LlamaSpec, train: TrainSpec, hw: HardwareSpec,
                  dtypes: DTypeBytes, adam_variant: str = "adamw") -> Dict[str, float]:
    # FLOPs
    F_fwd_step   = spec.flops_step_forward(train.batch_size, train.seq_len, train.grad_accum)
    F_train_step = spec.flops_step_train(train.batch_size, train.seq_len, train.grad_accum, train.checkpoint)
    F_adam_step  = spec.flops_step_adam(adam_variant)
    F_total      = F_train_step + F_adam_step

    # BYTES (weight I/O model for FW+BW + optimizer formula with dtypes)
    B_train_step = bytes_train_per_step(spec, train, dtypes)
    B_adam_step  = adam_bytes_per_step(spec, dtypes)
    B_total      = B_train_step + B_adam_step

    # Times = max(F/T, Bytes/B)
    t_fwbw_compute = F_train_step / hw.T
    t_fwbw_memory  = B_train_step / hw.B
    t_opt_compute  = F_adam_step  / hw.T
    t_opt_memory   = B_adam_step  / hw.B

    # Bottleneck decisions + human logs
    fwbw_label, fwbw_ratio, fwbw_msg = _which_bottleneck(t_fwbw_compute, t_fwbw_memory, "FW+BW")
    opt_label,  opt_ratio,  opt_msg  = _which_bottleneck(t_opt_compute,  t_opt_memory,  "Adam")

    # Roofline times
    t_fwbw = max(t_fwbw_compute, t_fwbw_memory)
    t_opt  = max(t_opt_compute,  t_opt_memory)
    t_step = t_fwbw + t_opt
    tps    = train.tokens_per_step / t_step
    
    print(fwbw_msg)
    print(opt_msg)
    if fwbw_label == "compute":
        print(f"FW+BW compute-bound: effective AI ≳ T/B. Consider raising arithmetic intensity (e.g., larger S) "
              f"or improving compute efficiency.")
    elif fwbw_label == "memory":
        print(f"FW+BW memory-bound: bytes dominate. Consider FlashAttention, fused kernels, or reducing traffic.")
    if opt_label == "memory":
        print(f"Adam is typically memory-bound (state reads/writes). Overlapping or sharding may help.")

    return {
        "F_fwd_step": F_fwd_step,
        "F_train_step": F_train_step,
        "F_adam_step": F_adam_step,
        "F_total_step": F_total,
        "B_fwd_train_step": B_train_step,
        "B_adam_step": B_adam_step,
        "B_total_step": B_total,
        "t_fwbw_compute_s": t_fwbw_compute,
        "t_fwbw_memory_s": t_fwbw_memory,
        "t_fwbw_s": t_fwbw,
        "t_opt_compute_s": t_opt_compute,
        "t_opt_memory_s": t_opt_memory,
        "t_opt_s": t_opt,
        "t_step_s": t_step,
        "tokens_per_step": train.tokens_per_step,
        "throughput_tok_per_s": tps,
    }

# ---------- memory footprint ----------
def memory_persistent_bytes(spec: LlamaSpec, dtypes: DTypeBytes) -> int:
    """
    Persistent model state:
      params (w) + grads (grad) + m + v
    """
    P = spec.total_params
    param_b = dtypes.w   * P
    grad_b  = dtypes.grad* P
    mv_b    = (dtypes.m + dtypes.v) * P
    return param_b + grad_b + mv_b

def memory_activation_bounds(spec: LlamaSpec, train: TrainSpec, dtypes: DTypeBytes) -> Tuple[int, int]:
    """
    Very rough bracket:
      lower (ckpt + FA): ~ L · (B·S) · (2D) · bytes(act)
      upper (no-ckpt):   ~ L · (B·S) · (3D + 2D_ff + 2D) · bytes(act)
    """
    M_micro = train.batch_size * train.seq_len
    bytes_per_el = dtypes.act
    lower = spec.num_hidden_layers * M_micro * (2*spec.hidden_size) * bytes_per_el if train.checkpoint else None
    upper = spec.num_hidden_layers * M_micro * (3*spec.hidden_size + 2*spec.intermediate_size + 2*spec.hidden_size) * bytes_per_el
    return lower, upper


# Print model spec
def print_out(spec, hw, train, dtypes):
    # Print shapes (one microbatch)
    print("\n=== Per-op GEMM shapes (microbatch) ===")
    for name, (A,B,C) in spec.gemm_shapes_one_microbatch(train.batch_size, train.seq_len).items():
        print(f"{name:24s}: {A} @ {B} -> {C}")

    # Params
    print("\n=== Parameters ===")
    print(f"Per-layer params   : {spec.per_layer_params:,}")
    print(f"Embedding params   : {spec.embedding_params:,}")
    print(f"LM head params     : {spec.lm_head_params:,}")
    print(f"TOTAL params       : {spec.total_params:,}  ({human_num(spec.total_params)})")

    # FLOPs (forward, train) and Adam
    F_tok = spec.flops_token_forward(train.seq_len)
    F_fwd = spec.flops_step_forward(train.batch_size, train.seq_len, train.grad_accum)
    F_trn = spec.flops_step_train(train.batch_size, train.seq_len, train.grad_accum, train.checkpoint)
    F_opt = spec.flops_step_adam("adam")
    print("\n=== FLOPs ===")
    print(f"Per-token forward  : {F_tok:,}  ({human_num(F_tok)})")
    print(f"Per-step forward   : {F_fwd:,}  ({human_num(F_fwd)})")
    print(f"Per-step training  : {F_trn:,}  ({human_num(F_trn)})")
    print(f"Per-step Adam      : {F_opt:,}  ({human_num(F_opt)})")

    # Bytes and roofline
    b_fwd_tok = bytes_forward_per_token(spec, train, dtypes)
    B_trn = bytes_train_per_step(spec, train, dtypes)
    B_opt = adam_bytes_per_step(spec, dtypes)
    res = roofline_step(spec, train, hw, dtypes, adam_variant="adam")

    print("\n=== Bytes (I/O) ===")
    print(f"Forward bytes/token: {human_bytes(b_fwd_tok)}")
    print(f"FW+BW bytes/step   : {human_bytes(B_trn)}")
    print(f"Adam bytes/step    : {human_bytes(B_opt)}")

    print("\n=== Roofline timing ===")
    print(f"FW+BW compute time : {res['t_fwbw_compute_s']:.4f} s")
    print(f"FW+BW memory time  : {res['t_fwbw_memory_s']:.4f} s")
    print(f"FW+BW roofline time: {res['t_fwbw_s']:.4f} s")
    print(f"Adam compute time  : {res['t_opt_compute_s']:.6f} s")
    print(f"Adam memory time   : {res['t_opt_memory_s']:.6f} s")
    print(f"Adam roofline time : {res['t_opt_s']:.6f} s")
    print(f"TOTAL step time    : {res['t_step_s']:.4f} s")
    print(f"Tokens/step        : {res['tokens_per_step']:,}")
    print(f"Throughput         : {res['throughput_tok_per_s']:.0f} tok/s ({human_num(res['throughput_tok_per_s'])})")

    # Memory footprint
    persist = memory_persistent_bytes(spec, dtypes)
    act_lo, act_hi = memory_activation_bounds(spec, train, dtypes)
    print("\n=== Memory footprint ===")
    print(f"Persistent (params+grads+m+v): {human_bytes(persist)}")
    if act_lo is not None:
        print(f"Activations lower (ckpt+FA)  : {human_bytes(act_lo)}")
    print(f"Activations upper (no-ckpt)  : {human_bytes(act_hi)}")
    print("\nDone.")

# ========================= example: LLaMA-3 8B, your HW =========================
if __name__ == "__main__":
    # LLaMA-3 8B config (matches your printout)
    spec8b = LlamaSpec(
        num_hidden_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,    # GQA
        intermediate_size=14336,
        vocab_size=128_256,
        tie_word_embeddings=False
    )
    # 3b
    spec = LlamaSpec(
        num_hidden_layers=28,
        hidden_size=3072,
        num_attention_heads=24,
        num_key_value_heads=8,    # GQA
        intermediate_size=8192,
        vocab_size=128_256,
        tie_word_embeddings=False
    )

    # -------- Hardware (PVC Max 1550 & A100) --------
    hw_pvc1550 = HardwareSpec(
        bandwidth_Bps_raw=1317e9, bandwidth_eff=1.0,   # searched 3276.8 GB/s HBM measured 1317e9
        compute_Fps_raw=232e12,  compute_eff=1.0      # searched 839 TFLOP/s  measured 232e12
    )
    # GPU 0: NVIDIA A100-SXM4-40GB
    hw_a100 = HardwareSpec(
        bandwidth_Bps_raw=1278e9, bandwidth_eff=1.0,   # searched 1555 GB/s peak  measured 1278e9
        compute_Fps_raw=229e12,  compute_eff=1.0       # searched 624 TFLOP/s  measured 229e12
    )

    # Dtypes (bf16 across the board → matches ~5909 tok/s case)
    dtypes = DTypeBytes(w=2, act=2, grad=2, m=2, v=2)

    # Training setup
    train = TrainSpec(
        batch_size=2,           # B
        seq_len=512,            # S
        grad_accum=1,           # G
        checkpoint=False,
        use_flash_attn=False,   # same behavior as your old flash_attention=False
        include_sdpa_bytes=False,
        include_act_io_linear=True,
        act_coeff_per_layer=2.0
    )

    print("----------on pvc max 1550------------")
    print_out(spec, hw_pvc1550, train, dtypes)
    print("----------on a100------------")
    print_out(spec, hw_a100, train, dtypes)
