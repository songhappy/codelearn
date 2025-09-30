import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from python.llama.llamaspec import LlamaSpec

# (Assumes your LlamaSpec dataclass from the prompt is already defined/imported)

# ===============================
# Utility: Rotary Positional Embeddings (RoPE)
# ===============================
def _rope_cos_sin(L: int, D: int, device, dtype, theta: float = 10000.0):
    """
    Build cos/sin tensors for RoPE.
    Shapes:
      inv_freq: (D/2,)
      t:        (L,)
      freqs:    (L, D/2)
      cos/sin:  (L, D)  (interleaved to match even/odd feature pairs)
    """
    assert D % 2 == 0, "RoPE requires even head_dim (D % 2 == 0)"
    inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, device=device, dtype=torch.float32) / D))
    t = torch.arange(L, device=device, dtype=torch.float32)
    freqs = torch.einsum("l,d->ld", t, inv_freq)  # (L, D/2)
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    # interleave along the last dim so we can broadcast directly to (..., D)
    cos = torch.repeat_interleave(cos, repeats=2, dim=-1)  # (L, D)
    sin = torch.repeat_interleave(sin, repeats=2, dim=-1)  # (L, D)
    return cos, sin

def _rope_rotate_half(x):
    """
    For x[..., 0::2], x[..., 1::2] (feature pairs), apply the 90° rotation:
      rot(x0, x1) = (-x1, x0)
    Shape preserved.
    """
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    out_even = -x_odd
    out_odd  =  x_even
    # interleave back to (..., D)
    out = torch.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out

def apply_rope_inplace(q: torch.Tensor, k: torch.Tensor, theta: float = 10000.0):
    """
    Apply RoPE to Q and K in-place (returns the same tensor objects for convenience).
    Expected shapes:
      q: (N, L_q, H, D)
      k: (N, L_k, H_kv, D)
    RoPE is applied independently to the (L_q, D) and (L_k, D) axes.
    """
    N, L_q, H, D = q.shape
    N2, L_k, H_kv, D2 = k.shape
    assert N == N2 and D == D2, "Q/K batch or head_dim mismatch for RoPE"

    # Build cos/sin for each sequence length separately
    cos_q, sin_q = _rope_cos_sin(L_q, D, device=q.device, dtype=q.dtype, theta=theta)  # (L_q, D)
    cos_k, sin_k = _rope_cos_sin(L_k, D, device=k.device, dtype=k.dtype, theta=theta)  # (L_k, D)

    # Broadcast to (1, L, 1, D)
    cos_q = cos_q.view(1, L_q, 1, D)
    sin_q = sin_q.view(1, L_q, 1, D)
    cos_k = cos_k.view(1, L_k, 1, D)
    sin_k = sin_k.view(1, L_k, 1, D)

    # q_rope = q * cos + rotate_half(q) * sin
    q.mul_(cos_q).add_(_rope_rotate_half(q), alpha=1.0).mul_(sin_q).add_(q, alpha=1.0)  # fused-ish work
    # The line above tries to reduce temps but is a bit dense; a clearer (equivalent) form is:
    # q[:] = q * cos_q + _rope_rotate_half(q) * sin_q

    # k_rope = k * cos + rotate_half(k) * sin
    k.mul_(cos_k).add_(_rope_rotate_half(k), alpha=1.0).mul_(sin_k).add_(k, alpha=1.0)
    # Similarly, a clearer (equivalent) form is:
    # k[:] = k * cos_k + _rope_rotate_half(k) * sin_k

    return q, k


# ===============================
# LlamaSpec-aligned Multi-Head (Self/Cross) Attention with GQA + RoPE
# ===============================
class LlamaMultiHeadAttention(nn.Module):
    """
    LlamaSpec-aligned attention:
      - H   = spec.num_attention_heads
      - H_kv= spec.num_key_value_heads (GQA: group size G = H / H_kv)
      - D   = spec.hidden_size
      - d_h = D / H
      - D_kv= H_kv * d_h

    Projections (no bias; HF-aligned):
      Wq: D×D,  Wk: D×D_kv,  Wv: D×D_kv,  Wo: D×D

    Supports:
      • Self-attention: pass the same tensor for (values, keys, query)
      • Cross-attention: different sources for (values/keys) vs query
      • GQA: replicate K/V heads across groups of Q heads
      • RoPE: rotary positional embedding on (Q, K)
    """
    def __init__(self, spec: LlamaSpec, rope_theta: float = 10000.0):
        super().__init__()
        self.spec = spec
        self.D   = spec.hidden_size
        self.H   = spec.num_attention_heads
        self.H_kv= spec.num_key_value_heads
        self.d_h = self.D // self.H
        self.D_kv= self.H_kv * self.d_h
        self.G   = self.H // self.H_kv  # group size for GQA
        self.rope_theta = rope_theta

        # --- validation (mirrors spec guards) ---
        assert self.D % self.H == 0, "hidden_size must be divisible by num_attention_heads"
        assert self.H % self.H_kv == 0, "num_attention_heads must be divisible by num_key_value_heads"
        assert self.d_h % 2 == 0, "RoPE expects even head_dim (d_h % 2 == 0)"

        # --- Linear maps for Q, K, V (no bias) ---
        # HF/Llama shapes:
        #   Q = X Wq ; Wq ∈ ℝ[D, D]           → Q_flat ∈ ℝ[M, D]
        #   K = X Wk ; Wk ∈ ℝ[D, D_kv]        → K_flat ∈ ℝ[M, D_kv]
        #   V = X Wv ; Wv ∈ ℝ[D, D_kv]        → V_flat ∈ ℝ[M, D_kv]
        self.q_proj = nn.Linear(self.D, self.D,    bias=False)
        self.k_proj = nn.Linear(self.D, self.D_kv, bias=False)
        self.v_proj = nn.Linear(self.D, self.D_kv, bias=False)

        # Output projection: Wo ∈ ℝ[D, D] (no bias)
        self.o_proj = nn.Linear(self.D, self.D, bias=False)

    def forward(self,
                values: torch.Tensor,
                keys:   torch.Tensor,
                query:  torch.Tensor,
                mask:   torch.Tensor = None,
                use_rope: bool = True):
        """
        Args:
            values: X_V ∈ ℝ[N, L_v, D]  (source for V)
            keys:   X_K ∈ ℝ[N, L_k, D]  (source for K)
            query:  X_Q ∈ ℝ[N, L_q, D]  (source for Q)
            mask:   Optional mask broadcastable to (N, H, L_q, L_k)
                    Boolean: True=keep, False=block; OR additive with -inf for blocked
            use_rope: apply RoPE to (Q, K) if True

        Returns:
            out: ℝ[N, L_q, D]
        
        Note: during training, L_q=L_k=L_v; during inference, L_k=L_v, L_k!=L_q
        """
        # --------- Dimensions ---------
        # N: batch size
        # L_v: value sequence length
        # L_k: key sequence length
        # L_q: query sequence length
        # D: model width
        # H: attention heads
        # H_kv: KV heads (GQA); G = H/H_kv
        # d_h: head dim D/H
        N = query.shape[0]
        L_v, L_k, L_q = values.shape[1], keys.shape[1], query.shape[1]

        # ============================================================
        # 1) Linear projections to get V, K, Q  (token-parallel GEMMs)
        # ============================================================
        # values @ Wv: (N, L_v, D) · (D, D_kv) → (N, L_v, D_kv)
        # keys   @ Wk: (N, L_k, D) · (D, D_kv) → (N, L_k, D_kv)
        # query  @ Wq: (N, L_q, D) · (D, D)    → (N, L_q, D)
        V_flat = self.v_proj(values)  # (N, L_v, D_kv)
        K_flat = self.k_proj(keys)    # (N, L_k, D_kv)
        Q_flat = self.q_proj(query)   # (N, L_q, D)

        # ============================================================
        # 2) Reshape into heads
        # ============================================================
        # Split last dims into heads with d_h = D/H:
        #   Q: (N, L_q, D)     -> (N, L_q, H, d_h)
        #   K: (N, L_k, D_kv)  -> (N, L_k, H_kv, d_h)
        #   V: (N, L_v, D_kv)  -> (N, L_v, H_kv, d_h)
        Q = Q_flat.view(N, L_q, self.H,   self.d_h)     # (N, L_q, H,   d_h)
        K = K_flat.view(N, L_k, self.H_kv, self.d_h)    # (N, L_k, H_kv, d_h)
        V = V_flat.view(N, L_v, self.H_kv, self.d_h)    # (N, L_v, H_kv, d_h)

        # ============================================================
        # 2.1) Apply RoPE to (Q, K) over (sequence, feature) dims
        # ============================================================
        # RoPE introduces relative positions by rotating pairs of features
        # with angle depending on token position.
        if use_rope:
            apply_rope_inplace(Q, K, theta=self.rope_theta)

        # ============================================================
        # 2.2) GQA: Map H query heads to H_kv KV heads by grouping
        # ============================================================
        # Group size G = H / H_kv (integer). Each group of G query heads
        # shares the same KV head. A convenient way is to repeat-interleave
        # K and V along head dimension to match H:
        #   K_gqa, V_gqa: (N, L_*, H, d_h)
        if self.G == 1:
            K_gqa = K  # (N, L_k, H, d_h)  [since H == H_kv]
            V_gqa = V  # (N, L_v, H, d_h)
        else:
            K_gqa = K.repeat_interleave(self.G, dim=2)  # (N, L_k, H_kv*G=H, d_h)
            V_gqa = V.repeat_interleave(self.G, dim=2)  # (N, L_v, H, d_h)

        # ============================================================
        # 3) Attention logits ("energy") via scaled dot-product
        # ============================================================
        # For each head h, query pos i, key pos j:
        #   score[i, j] = <Q[i, h, :], K[j, h, :]> / sqrt(d_h)
        # Matrix form (per head): Scores_h = Q_h K_h^T / sqrt(d_h)
        # Using einsum:
        #   Q: (N, L_q, H, d_h) -> "nqhd"
        #   K: (N, L_k, H, d_h) -> "nkhd"
        #   out: (N, H, L_q, L_k) -> "nhqk"
        energy = torch.einsum("nqhd,nkhd->nhqk", Q, K_gqa)  # (N, H, L_q, L_k)
        energy = energy / math.sqrt(self.d_h)              # scale by √d_h

        # ============================================================
        # 4) Apply mask (if provided)
        # ============================================================
        # Mask broadcastable to (N, H, L_q, L_k):
        #  - Boolean: True=keep, False=block → set blocked logits to -inf
        #  - Additive: add 0 for keep, -inf for block
        if mask is not None:
            if mask.dtype == torch.bool:
                energy = energy.masked_fill(~mask, float("-inf"))
            else:
                energy = energy + mask

        # ============================================================
        # 5) Softmax over keys to get attention distribution
        # ============================================================
        # For each query position, softmax over key positions:
        #   P = softmax(energy, dim=-1)  → (N, H, L_q, L_k)
        P = F.softmax(energy, dim=-1)

        # ============================================================
        # 6) Values aggregation (AV)
        # ============================================================
        # For each head, each query pos i:
        #   head_out[i] = Σ_j P[i, j] * V[j]
        # Einsum:
        #   P: (N, H, L_q, L_k) -> "nhqk"
        #   V: (N, L_v, H, d_h) -> "nlhd"  (assumes L_v == L_k for self-attn; common in cross-attn too)
        #   out: (N, L_q, H, d_h) -> "nqhd"
        out_heads = torch.einsum("nhqk,nkhd->nqhd", P, V_gqa)  # (N, L_q, H, d_h)

        # ============================================================
        # 7) Concatenate heads and final linear projection
        # ============================================================
        # Concat over head dim:
        #   (N, L_q, H, d_h) -> (N, L_q, H*d_h) = (N, L_q, D)
        out = out_heads.reshape(N, L_q, self.H * self.d_h)    # (N, L_q, D)
        # Output projection:
        #   Y = Concat(head_1..head_H) Wo ; Wo ∈ ℝ[D, D]
        out = self.o_proj(out)                                # (N, L_q, D)

        return out


# ===============================
# Helper: make a causal (autoregressive) mask
# ===============================
def make_causal_mask(L_q, L_k=None, device=None, dtype=torch.bool):
    """
    Create a lower-triangular mask that allows attending only to current and past positions.
    Returns shape (1, 1, L_q, L_k) so it broadcasts over (N, H, L_q, L_k).
    True = keep, False = block.
    """
    if L_k is None:
        L_k = L_q
    m = torch.ones((L_q, L_k), dtype=dtype, device=device).tril()
    return m.view(1, 1, L_q, L_k)


# ===============================
# Tiny sanity check / demo with a LlamaSpec (e.g., Llama 3.2 "3B"-ish)
# ===============================
if __name__ == "__main__":
    # Example spec (uncomment/adjust to your exact model)
    spec = LlamaSpec(
        num_hidden_layers=28,
        hidden_size=3072,          # D
        num_attention_heads=24,    # H
        num_key_value_heads=8,     # H_kv (GQA) → G = 3
        intermediate_size=8192,    # D_ff (not used here, but in MLP)
        vocab_size=128_256,
        tie_word_embeddings=False
    )

    torch.manual_seed(0)
    N, L_q, L_kv = 2, 512, 512
    D = spec.hidden_size

    x_q = torch.randn(N, L_q, D)
    x_kv = torch.randn(N, L_kv, D)

    attn = LlamaMultiHeadAttention(spec, rope_theta=10000.0)

    # Self-attention (Q=K=V=x), causal
    y_self = attn(values=x_kv, keys=x_kv, query=x_q, mask=make_causal_mask(L_q, L_kv, device=x_q.device))
    print("Self-attn (causal) out:", y_self.shape)  # (N, L_q, D)

    # Cross-attention demo (different Q vs KV sources)
    ctx = torch.randn(N, 7, D)
    y_cross = attn(values=ctx, keys=ctx, query=x_q, mask=None)
    print("Cross-attn out:", y_cross.shape)         # (N, L_q, D)
