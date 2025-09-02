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

# ---------- HF-config-aligned model spec with math ----------
@dataclass(frozen=True)
class LlamaSpec:
    """
    Math (↔ HF names):
      L := num_hidden_layers
      D := hidden_size
      H := num_attention_heads
      H_kv := num_key_value_heads
      D_ff := intermediate_size
      V := vocab_size
      d_h := D / H
      D_kv := H_kv · d_h
      r := D_ff / D

    Per-layer weights (no bias):
      Wq: D×D,  Wk: D×D_kv,  Wv: D×D_kv,  Wo: D×D,
      W1: D×D_ff, W3: D×D_ff, W2: D_ff×D

    Per-layer params:
      params_layer = 2·D² + 2·D·D_kv + 3·D·D_ff

    ---------------------------------- Matrix manipulation (forward) ----------------------------------
    Let X be the input hidden states for one layer.

    Shapes (microbatch with B sequences of length S):
      X ∈ ℝ[B, S, D]                        (flattened view: X_flat ∈ ℝ[M, D] with M = B·S)

    1) Linear projections (token-parallel GEMMs on the flattened token axis)
       Q_flat = X_flat @ Wq    ; Wq ∈ ℝ[D, D]        → Q_flat ∈ ℝ[M, D]
       K_flat = X_flat @ Wk    ; Wk ∈ ℝ[D, D_kv]     → K_flat ∈ ℝ[M, D_kv]
       V_flat = X_flat @ Wv    ; Wv ∈ ℝ[D, D_kv]     → V_flat ∈ ℝ[M, D_kv]

       Reshape to heads:
         Q = reshape(Q_flat, B, S, H, d_h)          → Q ∈ ℝ[B, S, H, d_h]
         K = reshape(K_flat, B, S, H_kv, d_h)       → K ∈ ℝ[B, S, H_kv, d_h]
         V = reshape(V_flat, B, S, H_kv, d_h)       → V ∈ ℝ[B, S, H_kv, d_h]

       (Grouped-Query Attention / GQA)
         Group size G = H / H_kv (integer). Each group of G query heads shares one K/V head.
         In einsum/broadcast form we index the KV head for query head h by floor(h / G).

       Positional encoding (RoPE)
         Apply RoPE to (Q,K) over the last two dims (S and d_h); typically implemented as complex rotations
         or angle mixing on pairs of features. (FLOPs small vs GEMMs; we ignore in FLOP totals.)

    2) Attention logits (QKᵀ) and softmax
       For each query head h, use its assigned KV head k(h) = ⌊h/G⌋:
         Scores[b, h, s_q, s_k] = ( Q[b, s_q, h, :] · K[b, s_k, k(h), :] ) / √d_h

       Shapes:
         per (b,h): Q[b,:,h,:] ∈ ℝ[S, d_h], K[b,:,k(h),:] ∈ ℝ[S, d_h]
         QKᵀ: (S, d_h) × (d_h, S) → (S, S)  ⇒ Scores ∈ ℝ[B, H, S, S]

       Softmax over the last dim (keys):
         P = softmax(Scores, dim = S_k)              → P ∈ ℝ[B, H, S, S]

       FLOPs (per sequence, per layer, all heads):
         QKᵀ: 2·S·S·d_h per head × H = 2·S²·D
         AV  : 2·S·S·d_h per head × H = 2·S²·D
         Total attention matmuls = 4·D·S² (per sequence, per layer).
         Per-token attention FLOPs = (4·D·S²) / S = 4·D·S.

    3) Values aggregation (AV)
       Context[b, h, s_q, :] = Σ_{s_k} P[b, h, s_q, s_k] · V[b, s_k, k(h), :]
       Shapes:
         P[b,h,:,:] ∈ ℝ[S, S],  V[b,:,k(h),:] ∈ ℝ[S, d_h]  → Context[b,h,:,:] ∈ ℝ[S, d_h]
       Stack heads and reshape:
         Concat(Context over h) → H_out ∈ ℝ[B, S, H·d_h] = ℝ[B, S, D]

    4) Output projection (token-parallel GEMM)
       Y_flat = reshape(H_out, M, D) @ Wo           ; Wo ∈ ℝ[D, D] → ℝ[M, D]
       Y = reshape(Y_flat, B, S, D)

    5) MLP (SwiGLU)
       A_flat = X_flat @ W1 ; W1 ∈ ℝ[D, D_ff]  → ℝ[M, D_ff]
       B_flat = X_flat @ W3 ; W3 ∈ ℝ[D, D_ff]  → ℝ[M, D_ff]
       Gating: G = SiLU(A_flat) ⊙ B_flat       → ℝ[M, D_ff]   (elementwise)
       Z_flat = G @ W2       ; W2 ∈ ℝ[D_ff, D] → ℝ[M, D]
       Reshape Z = ℝ[B, S, D]

    6) LM head (for loss)
       logits_flat = H_out_flat @ W_vocab ; W_vocab ∈ ℝ[D, V] → ℝ[M, V]
       (Weight tying removes a separate D×V parameter tensor but **not** this compute.)
       Cross-entropy uses logits and labels; softmax cost is small vs GEMMs and ignored in FLOPs.

    ---------------------------------- FLOPs summary ----------------------------------
    Per layer, per token (GEMMs only):
      • Projections + MLP:
          Q (2·D²) + K (2·D·D_kv) + V (2·D·D_kv) + O (2·D²) + W1 (2·D·D_ff) + W3 (2·D·D_ff) + W2 (2·D_ff·D)
          = 4·D² + 4·D·D_kv + 6·D·D_ff
      • Attention matmuls per token: 4·D·S
      • LM head per token: 2·D·V

    Forward FLOPs per token at length S:
      F_fwd,token(S) = L · [ (4·D² + 4·D·D_kv + 6·D·D_ff) + 4·D·S ] + 2·D·V

    Forward FLOPs per step (batch B, length S, accum G):
      M := B·S·G (tokens/step),   F_fwd,step = M · F_fwd,token(S)

    Training FLOPs per step (rule of thumb):
      F_train,step ≈ α · F_fwd,step,   with α≈4 (activation checkpointing) or α≈3 (no checkpointing)

    Adam/AdamW FLOPs per step:
      Per-parameter scalar op model:
        m_t = β1 m + (1-β1) g            → 3 ops
        v_t = β2 v + (1-β2) g²           → 4 ops
        bias correction (m̂, v̂)          → 2 ops
        denom = sqrt(v̂) + ε              → 2 ops
        ratio = m̂ / denom                → 1 op
        scale by lr + param update        → 2 ops
        AdamW decoupled weight decay      → +2 ops
      ⇒ Adam ≈ 14 ops/param, AdamW ≈ 16 ops/param
      F_adam,step = (ops/param) · params_total

    Notes:
      • All projection/MLP GEMMs are **token-parallel** across M=B·S·(microbatch), so we flatten tokens.
      • Attention is **sequence-quadratic**; GQA uses H_kv heads for K/V and broadcasts them over groups
        of size G=H/H_kv for Q, but overall FLOPs remain 4·D·S per token.
      • RoPE and elementwise activations/normalizations are omitted from FLOPs (small vs GEMMs) but
        still matter for memory traffic and kernel scheduling.
    """
    # HF names (verbatim)
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    tie_word_embeddings: bool = False  # removes D×V params, not compute

    # --- validation ---
    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    # --- derived dims ---
    @property
    def head_dim(self) -> int:      # d_h = D / H
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_dim_total(self) -> int:  # D_kv = H_kv · d_h
        return self.num_key_value_heads * self.head_dim

    @property
    def mlp_ratio(self) -> float:   # r = D_ff / D
        return self.intermediate_size / self.hidden_size

    # --- parameters (no bias) ---
    @property
    def per_layer_params(self) -> int:
        D, Dkv, Dff = self.hidden_size, self.kv_dim_total, self.intermediate_size
        return 2*D*D + 2*D*Dkv + 3*D*Dff

    @property
    def embedding_params(self) -> int:   # V·D
        return self.vocab_size * self.hidden_size

    @property
    def lm_head_params(self) -> int:     # D·V (0 if tied)
        return 0 if self.tie_word_embeddings else self.hidden_size * self.vocab_size

    @property
    def total_params(self) -> int:
        return (self.num_hidden_layers * self.per_layer_params
                + self.embedding_params + self.lm_head_params)

    # --- FLOPs ---
    def flops_token_forward(self, S: int) -> int:
        D, Dkv, Dff, L, V = self.hidden_size, self.kv_dim_total, self.intermediate_size, self.num_hidden_layers, self.vocab_size
        const = 4*D*D + 4*D*Dkv + 6*D*Dff
        attn  = 4*D*S
        vocab = 2*D*V
        return L * (const + attn) + vocab

    def flops_step_forward(self, B: int, S: int, G: int = 1) -> int:
        return (B * S * G) * self.flops_token_forward(S)

    def flops_step_train(self, B: int, S: int, G: int = 1, checkpoint: bool = False) -> int:
        alpha = 4 if checkpoint else 3
        return alpha * self.flops_step_forward(B, S, G)

    # --- Adam / AdamW FLOPs ---
    def adam_ops_per_param(self, variant: str = "adamw") -> int:
        # m:3, v:4, bias:2, sqrt+eps:2, div:1, lr+update:2, (AdamW decay:+2)
        ops = 3 + 4 + 2 + 2 + 1 + 2
        if variant.lower() == "adamw": ops += 2
        return ops  # Adam≈14, AdamW≈16

    def flops_step_adam(self, variant: str = "adamw") -> int:
        return self.adam_ops_per_param(variant) * self.total_params

    # --- GEMM shapes for one microbatch (B sequences, length S) ---
    def gemm_shapes_one_microbatch(self, B: int, S: int) -> Dict[str, Tuple[Tuple[int,int], Tuple[int,int], Tuple[int,int]]]:
        M_micro = B * S
        D, Dkv, Dff, V = self.hidden_size, self.kv_dim_total, self.intermediate_size, self.vocab_size
        shapes = {
            "Q = X Wq"            : ((M_micro, D),    (D, D),      (M_micro, D)),
            "K = X Wk"            : ((M_micro, D),    (D, Dkv),    (M_micro, Dkv)),
            "V = X Wv"            : ((M_micro, D),    (D, Dkv),    (M_micro, Dkv)),
            "O = Concat() Wo"     : ((M_micro, D),    (D, D),      (M_micro, D)),
            "A = X W1"            : ((M_micro, D),    (D, Dff),    (M_micro, Dff)),
            "B = X W3"            : ((M_micro, D),    (D, Dff),    (M_micro, Dff)),
            "Y = (...) W2"        : ((M_micro, Dff),  (Dff, D),    (M_micro, D)),
        }
        if not self.tie_word_embeddings:
            shapes["Logits = H W_vocab"] = ((M_micro, D), (D, V), (M_micro, V))
        return shapes

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
