# llama_project_train_ma.py (used)
# ---------------------------------------------------------------------------
# Training throughput projector (model-size methodology) for a LLaMA-3 8B–style model.
# Adds explicit SDPA (attention) compute with an alpha multiplier for training (FW+BW, ckpt).
# Weight I/O uses microbatch scaling (reads per FW+BW pass × grad_accum).
#
# NOTE on constants (detailed below where they are defined):
# • flop_factor_train = 6 (no ckpt) or 8 (with ckpt)
#     Derivation for a single linear Y = XW (MAC=2 FLOPs):
#       Forward:      ~2·M·K·N
#       Backward dX:  ~2·M·K·N
#       Backward dW:  ~2·M·K·N
#       -------------------------
#       Train (no ckpt): ~6·M·K·N  ⇒ ≈ 3× forward ⇒ per-param per-token ≈ 6
#       + Recompute FWD during BWD (activation checkpointing): +2·M·K·N
#       Train (ckpt):    ~8·M·K·N  ⇒ ≈ 4× forward ⇒ per-param per-token ≈ 8
#
# • alpha (for SDPA/attention): 3 (no ckpt) or 4 (with ckpt)
#     Attention cost is handled separately because it scales with sequence length S, not params.
#     We take Forward(attn) and multiply by:
#       α = 3  → FW + BW for attention (≈3× forward)
#       α = 4  → FW + BW + recompute forward with checkpointing (≈4× forward)
#
# • Weight read multiplier "2.0" in bytes model:
#     In one FW+BW pass, each weight tensor W is read once in Forward (Y=XW) and
#     once in Backward for dX (dX=dY·Wᵀ). The dW GEMM (dW=Xᵀ·dY) does not read W.
#     So minimal weight traffic ≈ 2 × P × bytes_per_weight, then × grad_accum (G).
#     If you want to model checkpointing re-forward reads, you could use 3× when ckpt,
#     but this script accounts checkpointing primarily in compute (flops), not bytes.
#
# PRINT NOTE:
#   We compute:
#       t_compute_total = (F_train_total + F_adam) / T
#       t_memory_total  = (bytes_fwbw + bytes_opt) / B
#   and print which is larger (with a ±2% tie band). FW+BW/Adam times are still shown
#   for reference, but no separate control lines for them.
# ---------------------------------------------------------------------------

from dataclasses import dataclass

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

# ---------------- Model / HW / Train specs ----------------
@dataclass(frozen=True)
class LlamaSpec:
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_dim_total(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def per_layer_params(self) -> int:
        D, Dkv, Dff = self.hidden_size, self.kv_dim_total, self.intermediate_size
        return 2*D*D + 2*D*Dkv + 3*D*Dff

    @property
    def embedding_params(self) -> int:
        return self.vocab_size * self.hidden_size

    @property
    def lm_head_params(self) -> int:
        return 0 if self.tie_word_embeddings else self.hidden_size * self.vocab_size

    @property
    def total_params(self) -> int:
        return (self.num_hidden_layers * self.per_layer_params
                + self.embedding_params + self.lm_head_params)

@dataclass(frozen=True)
class TrainSpec:
    batch_size: int     # B
    seq_len: int        # S
    grad_accum: int = 1 # G
    checkpoint: bool = False  # toggles alpha for SDPA term

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.seq_len * self.grad_accum

@dataclass(frozen=True)
class DTypeBytes:
    # DType sizes (bytes/element) used for coarse I/O accounting.
    w: int = 2   # weights dtype (bf16=2)
    act: int = 2
    grad: int = 2
    m: int = 2   # Adam moments in bf16 (use 4 for fp32)
    v: int = 2

@dataclass(frozen=True)
class HardwareSpec:
    bandwidth_Bps_raw: float
    bandwidth_eff: float
    compute_Fps_raw: float
    compute_eff: float

    @property
    def B(self) -> float:  # effective bandwidth (bytes/s)
        return self.bandwidth_Bps_raw * self.bandwidth_eff

    @property
    def T(self) -> float:  # effective compute (FLOPs/s)
        return self.compute_Fps_raw * self.compute_eff

# ---------------- Training projector (model-size + explicit SDPA) ----------------
@dataclass(frozen=True)
class ModelSizeTrainingConfig:
    # Param-proportional compute per param per token (covers proj/MLP/LM head FW+BW).
    # ≈6 (no checkpointing), ≈8 (with checkpointing recompute).
    flop_factor_train: float = 6.0
    # Adam ops per param (14 ~ Adam, 16 ~ AdamW)
    adam_ops_per_param: int = 16
    # Rough linear activation traffic term
    include_act_io_linear: bool = True
    act_coeff_per_layer: float = 2.0

    # ---- SDPA (attention) controls ----
    include_sdpa_compute: bool = True
    # alpha = training/FWD multiplier for SDPA (≈3 no-ckpt, ≈4 with ckpt)
    alpha_no_ckpt: float = 3.0
    alpha_ckpt: float  = 4.0
    # Optional SDPA bytes via arithmetic intensity heuristic
    include_sdpa_bytes: bool = False
    use_flash_attn: bool = True  # affects AI only if include_sdpa_bytes=True

def _alpha(train: TrainSpec, cfg: ModelSizeTrainingConfig) -> float:
    return cfg.alpha_ckpt if train.checkpoint else cfg.alpha_no_ckpt

def _ai_for_attention(S: int, use_flash: bool) -> float:
    # FLOPs/byte heuristic (same shape as in detailed script)
    return (S / 2.0) if use_flash else (82.0 + 0.005 * S)

def _which_control(t_compute: float, t_memory: float, tol: float = 0.02):
    """Return ('compute' | 'memory' | 'tied', ratio=compute/memory) with a ±2% tie band."""
    ratio = (t_compute + 1e-12) / (t_memory + 1e-12)
    if abs(ratio - 1.0) <= tol:
        return "tied", ratio
    return ("compute" if ratio > 1.0 else "memory"), ratio

def _compute_bytes_and_time(spec: LlamaSpec, train: TrainSpec, hw: HardwareSpec,
                            dtypes: DTypeBytes, cfg: ModelSizeTrainingConfig):
    """
    - F_param_train includes backward via cfg.flop_factor_train.
    - Adds explicit SDPA training compute: F_sdpa_train = alpha * L * (4*D*S) * (B*S*G).
    - Weight I/O is microbatch-scaled: reads per FW+BW pass × grad_accum.
    """
    
    P = spec.total_params
    L = spec.num_hidden_layers
    D = spec.hidden_size
    Bsz, S, G = train.batch_size, train.seq_len, train.grad_accum
    M = Bsz * S * G
    alpha = _alpha(train, cfg)

    # ---- Compute FLOPs ----
    # Param-proportional (proj/MLP/LM head). Set flop_factor_train≈6 (no-ckpt) or ≈8 (ckpt).
    F_param_train = cfg.flop_factor_train * P * M

    # Explicit SDPA training compute (scales with sequence length S)
    # per token per layer FWD ≈ 4*D*S; multiply by alpha for training cost
    F_sdpa_train = (alpha * L * 4.0 * D * S) * M if cfg.include_sdpa_compute else 0.0

    # Optimizer FLOPs
    F_adam = cfg.adam_ops_per_param * P

    # ---- Memory bytes ----
    # (See header note explaining the 2.0 weight read multiplier)
    bytes_weights_fw_bw = 2.0 * P * dtypes.w * G

    # Gradients written once per step (covers dW write traffic)
    bytes_grad = P * dtypes.grad

    # Adam read/write (bf16 moments; set m/v=4 for fp32 moments)
    #   read: W + g + m + v ; write: W + m + v
    bytes_opt = (P * (dtypes.w + dtypes.grad + dtypes.m + dtypes.v) +
                 P * (dtypes.w + dtypes.m + dtypes.v))

    # Coarse activation traffic (linear model for non-attention layers)
    bytes_act = 0
    if cfg.include_act_io_linear:
        bytes_act = M * L * (cfg.act_coeff_per_layer * D * dtypes.act)

    # Optional SDPA bytes via AI model (usually keep off in model-size mode)
    bytes_sdpa = 0
    if cfg.include_sdpa_bytes and cfg.include_sdpa_compute:
        AI = _ai_for_attention(S, cfg.use_flash_attn)  # FLOPs/byte
        bytes_sdpa = F_sdpa_train / max(AI, 1e-9)

    bytes_fwbw = bytes_weights_fw_bw + bytes_grad + bytes_act + bytes_sdpa
    bytes_total = bytes_fwbw + bytes_opt

    # ---- Roofline ----
    F_train_total = F_param_train + F_sdpa_train

    # split into compute vs memory components for each phase
    t_fwbw_compute = F_train_total / hw.T
    t_fwbw_memory  = bytes_fwbw   / hw.B
    t_opt_compute  = F_adam       / hw.T
    t_opt_memory   = bytes_opt    / hw.B

    # final step times
    t_fwbw = max(t_fwbw_compute, t_fwbw_memory)
    t_opt  = max(t_opt_compute,  t_opt_memory)
    t_step = t_fwbw + t_opt
    tps    = M / t_step

    # ----- SINGLE OVERALL CONTROL (compute vs memory) -----
    # Compare totals if everything were compute-bound vs memory-bound.
    t_compute_total = (F_train_total + F_adam) / hw.T
    t_memory_total  = (bytes_fwbw + bytes_opt) / hw.B
    overall_control, overall_ratio = _which_control(t_compute_total, t_memory_total)

    return {
        "params_total": P,
        "F_param_train": F_param_train,
        "F_sdpa_train": F_sdpa_train,
        "F_train_total": F_train_total,
        "F_adam_step": F_adam,
        "bytes_fwbw_step": bytes_fwbw,
        "bytes_opt_step": bytes_opt,
        "bytes_total_step": bytes_total,
        "t_fwbw_compute_s": t_fwbw_compute,
        "t_fwbw_memory_s":  t_fwbw_memory,
        "t_opt_compute_s":  t_opt_compute,
        "t_opt_memory_s":   t_opt_memory,
        "t_fwbw_s": t_fwbw,
        "t_opt_s": t_opt,
        "t_step_s": t_step,
        "tokens_per_step": M,
        "throughput_tok_per_s": tps,
        # Overall (single) control fields
        "t_compute_total_s": t_compute_total,
        "t_memory_total_s":  t_memory_total,
        "overall_control":   overall_control,
        "overall_ratio":     overall_ratio,
    }

def print_block(title: str, res):
    print(f"\n-- {title} --")
    print(f"Params total        : {res['params_total']:,} ({human_num(res['params_total'])})")
    print(f"F_param_train FLOPs : {human_num(res['F_param_train'])}")
    print(f"F_SDPA_train FLOPs  : {human_num(res['F_sdpa_train'])}")
    print(f"F_train_total FLOPs : {human_num(res['F_train_total'])}")
    print(f"Adam FLOPs/step     : {human_num(res['F_adam_step'])}")
    print(f"FW+BW bytes/step    : {human_bytes(res['bytes_fwbw_step'])}")
    print(f"Adam bytes/step     : {human_bytes(res['bytes_opt_step'])}")
    print(f"TOTAL bytes/step    : {human_bytes(res['bytes_total_step'])}")
    # SINGLE overall control printout (no separate FW+BW/Adam control lines)
    print(f"Control             : {res['overall_control']} "
          f"(compute_total={res['t_compute_total_s']:.4f}s, memory_total={res['t_memory_total_s']:.4f}s, "
          f"ratio={res['overall_ratio']:.2f})")
    # Phase times still shown for context
    print(f"FW+BW time          : {res['t_fwbw_s']:.4f} s")
    print(f"Adam time           : {res['t_opt_s']:.4f} s")
    print(f"TOTAL step time     : {res['t_step_s']:.4f} s")
    print(f"Tokens/step         : {res['tokens_per_step']:,}")
    print(f"Throughput          : {res['throughput_tok_per_s']:.0f} tok/s ({human_num(res['throughput_tok_per_s'])})")

def run():
    # -------- LLaMA-3 8B–like config (matches your module printout) --------
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

    # -------- Training setup --------
    train = TrainSpec(batch_size=2, seq_len=512, grad_accum=1, checkpoint=False)

    dtypes = DTypeBytes(w=2, act=2, grad=2, m=2, v=2)  # bf16 states everywhere

    # If you checkpoint, set checkpoint=True and bump flop_factor_train≈8.0
    cfg = ModelSizeTrainingConfig(
        flop_factor_train=6.0,     # no-ckpt; use 8.0 if train.checkpoint=True
        adam_ops_per_param=16,
        include_act_io_linear=True,
        act_coeff_per_layer=2.0,
        include_sdpa_compute=True, # <— SDPA compute ON
        include_sdpa_bytes=False,  # keep False for coarse model-size mode
        use_flash_attn=True
    )

    print("\n================ TRAINING THROUGHPUT (model-size + explicit SDPA) ================")
    for name, hw in [("PVC Max 1550", hw_pvc1550), ("A100 40GB", hw_a100)]:
        print(f"\n### {name}  (B={train.batch_size}, S={train.seq_len}, G={train.grad_accum}, ckpt={train.checkpoint})")
        res = _compute_bytes_and_time(spec, train, hw, dtypes, cfg)
        print_block("Model-size + SDPA", res)

if __name__ == "__main__":
    run()
