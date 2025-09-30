"""
Attention cheat sheet
1) MultiHeadAttention, 
2) MultiHeadGroupedAttention, 
3) FlashAttention, 
4) MultiHeadFlexAttention, 
5) MultiHeadPagedAttention,
6) MultiHeadLatentAttention
---------------------------------------

GQA is a version of multi-headed attention (MHA) which uses fewer key/value
heads than query heads by grouping n query heads for each key and value head.
Multi-Query Attention (MQA) is the extreme case with a single key/value head
shared by all query heads.

(credit for the documentation:
`litgpt.Config <https://github.com/Lightning-AI/litgpt/blob/eda1aaaf391fd689664f95487ab03dc137e213fd/litgpt/config.py>`_)

::

    ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
    └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    │    │    │    │         │        │                 │
    ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
    └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
    ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
    │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
    └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
    ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
            MHA                    GQA                   MQA
    n_kv_heads =4          n_kv_heads=2           n_kv_heads=1
"""

# =====================================================================
# Overview: Expanded vs Grouped compute (and why grouped can be faster)
#
# Notation:
#   D    = hidden_size (model width)
#   H    = num_attention_heads
#   d_h  = D // H  (per-head dimension)
#   H_kv = num_key_value_heads
#   G    = H // H_kv  (Q heads per K/V head)
#   D_kv = H_kv * d_h
# 
# With identical weights, both variants compute the SAME attention math:
#   P = softmax(Q K^T / sqrt(d_h)),  O = P V
# The difference is *how* K,V are handled when H_kv < H.
#
# Expanded KV (class MultiHeadAttention):
#   - Duplicate K,V heads so K,V have H heads (repeat each KV head G = H/H_kv times).
#     e.g., Llama 3.2 3B: H = 24, H_kv = 8  →  G = 3  (D = 3072 ⇒ d_h = 128, D_kv = 1024)
#   - Simple and Flash/SDPA-friendly, shapes after reshape: Q,K,V -> [N, L, H, d_h].
#   - But it reads/writes G× more K/V data (extra memory traffic).
#
# Grouped KV (class MultiHeadGroupedAttention):
#   - Do NOT duplicate K,V. Keep K,V at H_kv heads; treat Q as (H_kv, G) groups.
#   - Reuses the same K,V tiles across G Q-heads → far less memory traffic.
#   - Needs grouped-aware math (we do it via einsum here).
#
# Where the speedup comes from (grouped):
#   1) No KV expansion inside attention → up to G× less KV bandwidth.
#   2) Cheaper K/V projections and fewer params: Wk, Wv ∈ R[D, H_kv·d_h] vs R[D, H·d_h].
#   3) Much smaller KV cache at inference (store/read [H_kv, L, d_h] vs [H, L, d_h]).
#
# We also include:
#   • FlashAttention: I/O-aware, fused kernel via SDPA that tiles Q,K,V in SRAM
#     so the [L_q, L_k] attention matrix is never materialized (O(n) memory).
#   • FlexAttention: PyTorch fused attention that supports GQA directly
#     (no KV expansion), plus extensible score/mask mods.
#   • PagedAttention: inference-time KV cache in fixed-size pages; reduces
#     copies/fragmentation and plays nicely with allocator & scheduler.
#   • LatentAttention: 2-stage bottleneck with L_m << {L_q, L_k} to reduce
#     attention matrix sizes and I/O.
# =====================================================================

from typing import Dict, Tuple, Optional, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from python.llama.llamaspec import LlamaSpec

# Optional backends
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend  # FlashAttention via SDPA
    HAS_SDPA = True
except Exception:
    HAS_SDPA = False

try:
    from torch.nn.attention.flex_attention import flex_attention as flex_attn  # FlexAttention
    _FLEX_IMPORT_OK = True
except Exception:
    flex_attn = None
    _FLEX_IMPORT_OK = False

# Only advertise Flex if PyTorch was built with CUDA and a CUDA device is available
HAS_FLEX = bool(_FLEX_IMPORT_OK and torch.backends.cuda.is_built() and torch.cuda.is_available())


# ------------------------- small utilities -------------------------

def _stable_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    return F.softmax(x - x.max(dim=dim, keepdim=True).values, dim=dim)


# ===================================================================
# 1) Expanded KV: standard MHA path (simple, but duplicates KV)
# ===================================================================

class MultiHeadAttention(nn.Module):
    """
    Standard MHA with optional H_kv < H by *expanding* K/V heads to H.
    No mask, no RoPE. Cross-attn allowed; requires L_k == L_v.

    Why this is simple but heavier:
      • We replicate each KV head G = H/H_kv times so K,V have H heads.
      • Pros: Drop-in with PyTorch SDPA/Flash; simplest code path.
      • Cons: G× more KV memory traffic (and larger KV cache at inference).

    Shapes:
      D=hidden, H=num_attention_heads, H_kv=num_key_value_heads, d_h=D//H, D_kv=H_kv*d_h
      Q: (N, L_q, H, d_h)
      K: (N, L_k, H, d_h)  # after expansion if H_kv < H
      V: (N, L_k, H, d_h)
    """
    def __init__(self, spec: LlamaSpec):
        super().__init__()
        self.D    = spec.hidden_size
        self.H    = spec.num_attention_heads
        self.H_kv = spec.num_key_value_heads
        assert self.D % self.H == 0
        assert self.H % self.H_kv == 0
        self.d_h  = self.D // self.H
        self.D_kv = self.H_kv * self.d_h
        self.G    = self.H // self.H_kv

        self.q_proj = nn.Linear(self.D, self.D,    bias=False)
        self.k_proj = nn.Linear(self.D, self.D_kv, bias=False)
        self.v_proj = nn.Linear(self.D, self.D_kv, bias=False)
        self.o_proj = nn.Linear(self.D, self.D,    bias=False)

    def forward(self, values: torch.Tensor, keys: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        N, L_v, _ = values.shape
        _, L_k, _ = keys.shape
        _, L_q, _ = query.shape
        if L_v != L_k:
            raise ValueError(f"L_v ({L_v}) must equal L_k ({L_k}).")

        Q = self.q_proj(query ).view(N, L_q, self.H,    self.d_h)
        K = self.k_proj(keys  ).view(N, L_k, self.H_kv, self.d_h)
        V = self.v_proj(values).view(N, L_k, self.H_kv, self.d_h)

        if self.H_kv != self.H:
            K = K.unsqueeze(3).expand(N, L_k, self.H_kv, self.G, self.d_h).reshape(N, L_k, self.H, self.d_h)
            V = V.unsqueeze(3).expand(N, L_k, self.H_kv, self.G, self.d_h).reshape(N, L_k, self.H, self.d_h)

        energy = torch.einsum("nqhd,nkhd->nhqk", Q, K) / math.sqrt(self.d_h)
        P = _stable_softmax(energy, dim=-1)
        out_heads = torch.einsum("nhqk,nkhd->nqhd", P, V)

        out = out_heads.reshape(N, L_q, self.H * self.d_h)
        return self.o_proj(out)


# ===================================================================
# 2) Grouped KV (GQA): no KV expansion, reuse KV across Q groups
# ===================================================================

class MultiHeadGroupedAttention(nn.Module):
    """
    Grouped compute (true GQA) with H_kv < H and *no* KV expansion.
    No mask, no RoPE. Cross-attn allowed; requires L_k == L_v.

    Why faster/leaner:
      • Avoids materializing repeated KV → up to Gx less KV bandwidth.
      • Smaller K/V projections & fewer params (D×H_kv*d vs D×H*d).
      • Much smaller KV cache at inference.

    Shapes:
      G = H // H_kv
      Q: (N, L_q, H,    d_h) → view (N, L_q, H_kv, G, d_h)
      K: (N, L_k, H_kv, d_h)
      V: (N, L_k, H_kv, d_h)
    """
    def __init__(self, spec: LlamaSpec):
        super().__init__()
        self.D    = spec.hidden_size
        self.H    = spec.num_attention_heads
        self.H_kv = spec.num_key_value_heads
        assert self.D % self.H == 0
        assert self.H % self.H_kv == 0
        self.d_h  = self.D // self.H
        self.D_kv = self.H_kv * self.d_h
        self.G    = self.H // self.H_kv

        self.q_proj = nn.Linear(self.D, self.D,    bias=False)
        self.k_proj = nn.Linear(self.D, self.D_kv, bias=False)
        self.v_proj = nn.Linear(self.D, self.D_kv, bias=False)
        self.o_proj = nn.Linear(self.D, self.D,    bias=False)

    def forward(self, values: torch.Tensor, keys: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        N, L_v, _ = values.shape
        _, L_k, _ = keys.shape
        _, L_q, _ = query.shape
        if L_v != L_k:
            raise ValueError(f"L_v ({L_v}) must equal L_k ({L_k}).")

        V = self.v_proj(values).view(N, L_k, self.H_kv, self.d_h)
        K = self.k_proj(keys  ).view(N, L_k, self.H_kv, self.d_h)
        Q = self.q_proj(query ).view(N, L_q, self.H,    self.d_h)

        Qg = Q.view(N, L_q, self.H_kv, self.G, self.d_h)
        energy_g = torch.einsum("nqhgd,nkhd->nhgqk", Qg, K) / math.sqrt(self.d_h)
        P_g = _stable_softmax(energy_g, dim=-1)

        out_g = torch.einsum("nhgqk,nkhd->nqhgd", P_g, V)           # (N, L_q, H_kv, G, d)
        out_heads = out_g.reshape(N, L_q, self.H, self.d_h)
        out = out_heads.reshape(N, L_q, self.H * self.d_h)
        return self.o_proj(out)


# ===================================================================
# 3) FlashAttention (via PyTorch SDPA): IO-aware fused kernels
# ===================================================================

class MultiHeadFlashAttention(MultiHeadAttention):
    """
    FlashAttention path using PyTorch SDPA's FLASH_ATTENTION backend.
    Inherits Expanded KV path (we expand K/V → H) because SDPA expects
    H_q == H_k == H_v.

    Why it's faster:
      • Fused kernel that *tiles* Q,K,V in SRAM and never materializes
        the full [L_q, L_k] attention matrix → O(n) memory, IO-aware.
      • Fewer kernel launches + better cache behavior.

    Notes:
      • Requires CUDA and a supported dtype to actually hit Flash; we
        request it via `sdpa_kernel([SDPBackend.FLASH_ATTENTION])`.
      • Falls back gracefully if unavailable.
    """
    def forward(self, values: torch.Tensor, keys: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        if not HAS_SDPA:
            return super().forward(values, keys, query)

        N, L_v, _ = values.shape
        _, L_k, _ = keys.shape
        _, L_q, _ = query.shape
        if L_v != L_k:
            raise ValueError(f"L_v ({L_v}) must equal L_k ({L_k}).")

        Q = self.q_proj(query ).view(N, L_q, self.H,    self.d_h)
        K = self.k_proj(keys  ).view(N, L_k, self.H_kv, self.d_h)
        V = self.v_proj(values).view(N, L_k, self.H_kv, self.d_h)

        if self.H_kv != self.H:
            K = K.unsqueeze(3).expand(N, L_k, self.H_kv, self.G, self.d_h).reshape(N, L_k, self.H, self.d_h)
            V = V.unsqueeze(3).expand(N, L_k, self.H_kv, self.G, self.d_h).reshape(N, L_k, self.H, self.d_h)

        Qb = Q.permute(0, 2, 1, 3).contiguous()
        Kb = K.permute(0, 2, 1, 3).contiguous()
        Vb = V.permute(0, 2, 1, 3).contiguous()
        scale = self.d_h ** -0.5

        try:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                O = F.scaled_dot_product_attention(Qb, Kb, Vb, attn_mask=None, dropout_p=0.0, scale=scale)
        except Exception:
            O = F.scaled_dot_product_attention(Qb, Kb, Vb, attn_mask=None, dropout_p=0.0, scale=scale)

        out = O.permute(0, 2, 1, 3).contiguous().view(N, L_q, self.H * self.d_h)
        return self.o_proj(out)


# ===================================================================
# 4) FlexAttention (PyTorch): flexible fused kernel with GQA support
# ===================================================================

class MultiHeadFlexAttention(nn.Module):
    """
    FlexAttention backend (grouped-aware). No mask, no RoPE; cross-attn ok.

    Why it's faster/flexible:
      • Fused attention with composable score modifiers and masking.
      • Supports H_kv != H *without* KV expansion (enable_gqa=True),
        so we keep GQA's KV bandwidth win *inside* the kernel.

    Falls back to grouped-einsum if FlexAttention is not available.
    """
    def __init__(self, spec: LlamaSpec):
        super().__init__()
        self.D    = spec.hidden_size
        self.H    = spec.num_attention_heads
        self.H_kv = spec.num_key_value_heads
        assert self.D % self.H == 0 and self.H % self.H_kv == 0
        self.d_h  = self.D // self.H
        self.D_kv = self.H_kv * self.d_h

        self.q_proj = nn.Linear(self.D, self.D,    bias=False)
        self.k_proj = nn.Linear(self.D, self.D_kv, bias=False)
        self.v_proj = nn.Linear(self.D, self.D_kv, bias=False)
        self.o_proj = nn.Linear(self.D, self.D,    bias=False)

    def forward(self, values: torch.Tensor, keys: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        N, L_v, _ = values.shape
        _, L_k, _ = keys.shape
        _, L_q, _ = query.shape
        if L_v != L_k:
            raise ValueError(f"L_v ({L_v}) must equal L_k ({L_k}).")

        V = self.v_proj(values).view(N, L_k, self.H_kv, self.d_h)
        K = self.k_proj(keys  ).view(N, L_k, self.H_kv, self.d_h)
        Q = self.q_proj(query ).view(N, L_q, self.H,    self.d_h)
        scale = self.d_h ** -0.5

        if HAS_FLEX:
            Qb = Q.permute(0, 2, 1, 3).contiguous()        # (N, H,   L_q, d)
            Kb = K.permute(0, 2, 1, 3).contiguous()        # (N, H_kv, L_k, d)
            Vb = V.permute(0, 2, 1, 3).contiguous()        # (N, H_kv, L_k, d)
            Ob = flex_attn(Qb, Kb, Vb, score_mod=None, block_mask=None, scale=scale, enable_gqa=True)
            out = Ob.permute(0, 2, 1, 3).contiguous().view(N, L_q, self.H * self.d_h)
            return self.o_proj(out)

        # Fallback: grouped-einsum
        H, H_kv, d = self.H, self.H_kv, self.d_h
        G = H // H_kv
        Qg = Q.view(N, L_q, H_kv, G, d)
        energy_g = torch.einsum("nqhgd,nkhd->nhgqk", Qg, K) * scale
        P_g = _stable_softmax(energy_g, dim=-1)
        out_g = torch.einsum("nhgqk,nkhd->nqhgd", P_g, V)
        out_heads = out_g.reshape(N, L_q, H, d)
        out = out_heads.reshape(N, L_q, H * d)
        return self.o_proj(out)


# ===================================================================
# 5) PagedAttention: paged KV cache for decode-time efficiency
# ===================================================================

class PagedKVCache:
    """
    Minimal paged KV cache (inspired by vLLM):
      • Organizes per-sequence K/V into fixed-size pages [P, S, H_kv, d],
        where S=page_size, P grows with length.
      • Appends write into pages without moving old data.
      • Saves space/time by avoiding large reallocations/copies and reducing
        fragmentation; enables sharing/compaction strategies.

    Educational cache (no eviction/compaction policies).
    """
    def __init__(self, batch_size: int, H_kv: int, d: int, page_size: int, device=None, dtype=None):
        self.B = batch_size
        self.H_kv = H_kv
        self.d = d
        self.S = page_size
        self.device = device
        self.dtype = dtype

        self.K_pages: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        self.V_pages: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        self.lengths = [0 for _ in range(batch_size)]

    def append(self, b: int, K_new: torch.Tensor, V_new: torch.Tensor):
        """
        Append (T_new, H_kv, d) tokens to sequence b.
        Splits across as many pages as needed.
        """
        assert K_new.shape == V_new.shape and K_new.ndim == 3
        assert K_new.shape[1] == self.H_kv and K_new.shape[2] == self.d
        t = K_new.shape[0]
        off = 0
        while off < t:
            # allocate a new page if needed
            if len(self.K_pages[b]) == 0 or self.lengths[b] % self.S == 0:
                self.K_pages[b].append(torch.empty(self.S, self.H_kv, self.d, device=self.device, dtype=self.dtype))
                self.V_pages[b].append(torch.empty(self.S, self.H_kv, self.d, device=self.device, dtype=self.dtype))

            page_pos = self.lengths[b] % self.S
            take = min(self.S - page_pos, t - off)
            self.K_pages[b][-1][page_pos:page_pos+take].copy_(K_new[off:off+take])
            self.V_pages[b][-1][page_pos:page_pos+take].copy_(V_new[off:off+take])

            self.lengths[b] += take
            off += take

    def materialize(self, b: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenate pages for sequence b to a single (L_k, H_kv, d) view.
        In production you'd iterate pages in the kernel (no concat).
        """
        if self.lengths[b] == 0:
            empty = torch.empty(0, self.H_kv, self.d, device=self.device, dtype=self.dtype)
            return empty, empty
        K = torch.cat([pg for pg in self.K_pages[b]], dim=0)[: self.lengths[b]]
        V = torch.cat([pg for pg in self.V_pages[b]], dim=0)[: self.lengths[b]]
        return K, V


class MultiHeadPagedAttention(nn.Module):
    """
    Grouped attention that *reads K/V from a paged cache* (no KV expansion).
    Designed for decode-time usage where keys/values accumulate over time.

    Why it saves space/time:
      • KV cache stored in fixed-size pages → minimal copying when appending,
        less fragmentation, and easier sharing/compaction → better memory use.
      • Attention still benefits from GQA: K/V stored as H_kv heads, not H.

    API:
      - If (values, keys) are passed, we both ATTEND and APPEND them to the cache.
      - If values/keys=None, we ATTEND to the existing cache only (typical decode step).
    """
    def __init__(self, spec: LlamaSpec, page_size: int = 128):
        super().__init__()
        self.D    = spec.hidden_size
        self.H    = spec.num_attention_heads
        self.H_kv = spec.num_key_value_heads
        assert self.D % self.H == 0 and self.H % self.H_kv == 0
        self.d_h  = self.D // self.H
        self.D_kv = self.H_kv * self.d_h
        self.G    = self.H // self.H_kv
        self.page_size = page_size

        self.q_proj = nn.Linear(self.D, self.D,    bias=False)
        self.k_proj = nn.Linear(self.D, self.D_kv, bias=False)
        self.v_proj = nn.Linear(self.D, self.D_kv, bias=False)
        self.o_proj = nn.Linear(self.D, self.D,    bias=False)

        self.cache: Optional[PagedKVCache] = None  # lazily created

    def forward(self,
                values: Optional[torch.Tensor],
                keys:   Optional[torch.Tensor],
                query:  torch.Tensor) -> torch.Tensor:
        N, L_q, _ = query.shape
        device, dtype = query.device, query.dtype
        if self.cache is None:
            self.cache = PagedKVCache(batch_size=N, H_kv=self.H_kv, d=self.d_h,
                                      page_size=self.page_size, device=device, dtype=dtype)

        # Append new K/V if provided (prefill or a decode step with new tokens)
        if values is not None and keys is not None:
            assert values.shape[0] == N and keys.shape[0] == N and values.shape[1] == keys.shape[1]
            L_k = values.shape[1]
            K_proj = self.k_proj(keys  ).view(N, L_k, self.H_kv, self.d_h)
            V_proj = self.v_proj(values).view(N, L_k, self.H_kv, self.d_h)
            for b in range(N):
                self.cache.append(b, K_proj[b], V_proj[b])

        # Materialize (demo). Production kernels iterate pages directly.
        Ks: List[torch.Tensor] = []
        Vs: List[torch.Tensor] = []
        Ls: List[int] = []
        for b in range(N):
            Kb, Vb = self.cache.materialize(b)             # (L_b, H_kv, d)
            Ks.append(Kb)
            Vs.append(Vb)
            Ls.append(Kb.shape[0])
        if any(l == 0 for l in Ls):
            # nothing to attend to
            return self.o_proj(self.q_proj(query))

        L_k_max = max(Ls)

        # Pad each seq to common length for batched einsum (simple demo approach)
        def pad_to(x: torch.Tensor, L: int) -> torch.Tensor:
            if x.shape[0] == L: return x
            pad = torch.zeros(L - x.shape[0], *x.shape[1:], device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=0)

        K = torch.stack([pad_to(k, L_k_max) for k in Ks], dim=0)   # (N, L_k_max, H_kv, d)
        V = torch.stack([pad_to(v, L_k_max) for v in Vs], dim=0)   # (N, L_k_max, H_kv, d)
        Q = self.q_proj(query).view(N, L_q, self.H, self.d_h)

        # Grouped attention over cached KV (no expansion)
        Qg = Q.view(N, L_q, self.H_kv, self.G, self.d_h)
        energy_g = torch.einsum("nqhgd,nkhd->nhgqk", Qg, K) / math.sqrt(self.d_h)
        P_g = _stable_softmax(energy_g, dim=-1)
        out_g = torch.einsum("nhgqk,nkhd->nqhgd", P_g, V)

        out_heads = out_g.reshape(N, L_q, self.H, self.d_h)
        out = out_heads.reshape(N, L_q, self.H * self.d_h)
        return self.o_proj(out)


# ===================================================================
# 6) Latent-bottleneck attention (Perceiver-style 2-stage)
# ===================================================================

class MultiHeadLatentAttention(nn.Module):
    """
    Latent-bottleneck multi-head attention (2 stages). No mask, no RoPE.
    Cross-attn allowed; requires L_k == L_v.

    Why smaller/faster (when L_m << min(L_q, L_k)):
      • Standard MHA cost:  O(H * L_q * L_k * d_h), attention matrix [H, L_q, L_k].
      • Latent MHA cost:    O(H * (L_m * L_k + L_q * L_m) * d_h)
          Stage A (compress):     [H, L_m, L_k]
          Stage B (query latents): [H, L_q, L_m]
      Memory/IO and softmax scale with L_m (small), not L_q or L_k.
      At inference, you can precompute/cache Stage-A latents Z (size L_m)
      for the prefix and only attend to them during decode → much less KV
      cache traffic than full K/V.

    Shapes:
      D = spec.hidden_size, H = spec.num_attention_heads
      H_kv = spec.num_key_value_heads, d_h = D // H, D_kv = H_kv * d_h
      L_m = latent_size
      Stage A: Q_lat=[N, L_m, H, d],  K/V_in=[N, L_k, H, d] (after expand)
      Stage B: Q_in =[N, L_q, H, d],  K/V_lat=[N, L_m, H, d] (after expand)
    """
    def __init__(self, spec: LlamaSpec, latent_size: int = 32, init_scale: float = 0.02):
        super().__init__()
        self.D    = spec.hidden_size
        self.H    = spec.num_attention_heads
        self.H_kv = spec.num_key_value_heads
        assert self.D % self.H == 0
        assert self.H % self.H_kv == 0
        self.d_h  = self.D // self.H
        self.D_kv = self.H_kv * self.d_h
        self.G    = self.H // self.H_kv
        self.L_m  = latent_size

        # Learned latent queries (shared across batch)
        self.latents = nn.Parameter(torch.randn(self.L_m, self.D) * init_scale)

        # Reuse a single set of projections in both stages
        self.q_proj = nn.Linear(self.D, self.D,    bias=False)
        self.k_proj = nn.Linear(self.D, self.D_kv, bias=False)
        self.v_proj = nn.Linear(self.D, self.D_kv, bias=False)
        self.o_proj = nn.Linear(self.D, self.D,    bias=False)

    def _expand_kv(self, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.H_kv == self.H:
            return K, V
        N, L, H_kv, d = K.shape
        K = K.unsqueeze(3).expand(N, L, H_kv, self.G, d).reshape(N, L, self.H, d)
        V = V.unsqueeze(3).expand(N, L, H_kv, self.G, d).reshape(N, L, self.H, d)
        return K, V

    def forward(self, values: torch.Tensor, keys: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        N, L_v, _ = values.shape
        _, L_k, _ = keys.shape
        _, L_q, _ = query.shape
        if L_v != L_k:
            raise ValueError(f"L_v ({L_v}) must equal L_k ({L_k}).")

        # Stage A: latents attend to the inputs → Z (N, L_m, D)
        latent = self.latents.unsqueeze(0).expand(N, self.L_m, self.D)
        Qm = self.q_proj(latent).view(N, self.L_m, self.H,    self.d_h)
        K1 = self.k_proj(keys  ).view(N, L_k,      self.H_kv, self.d_h)
        V1 = self.v_proj(values).view(N, L_k,      self.H_kv, self.d_h)
        K1e, V1e = self._expand_kv(K1, V1)  # → (N, L_k, H, d)

        E1 = torch.einsum("nqhd,nkhd->nhqk", Qm, K1e) / math.sqrt(self.d_h)
        P1 = _stable_softmax(E1, dim=-1)
        Z_heads = torch.einsum("nhqk,nkhd->nqhd", P1, V1e)                  # (N, L_m, H, d)
        Z = Z_heads.reshape(N, self.L_m, self.H * self.d_h)                 # (N, L_m, D)

        # Stage B: queries attend to latents Z
        Q2 = self.q_proj(query).view(N, L_q, self.H, self.d_h)
        K2 = self.k_proj(Z    ).view(N, self.L_m, self.H_kv, self.d_h)
        V2 = self.v_proj(Z    ).view(N, self.L_m, self.H_kv, self.d_h)
        K2e, V2e = self._expand_kv(K2, V2)                                  # → (N, L_m, H, d)

        E2 = torch.einsum("nqhd,nkhd->nhqk", Q2, K2e) / math.sqrt(self.d_h)
        P2 = _stable_softmax(E2, dim=-1)
        out_heads = torch.einsum("nhqk,nkhd->nqhd", P2, V2e)                # (N, L_q, H, d)

        out = out_heads.reshape(N, L_q, self.H * self.d_h)
        return self.o_proj(out)


# ------------------------- shapes printer & demo -------------------------

def print_projection_shapes(N: int, L_q: int, L_k: int, D: int, H: int, H_kv: int) -> None:
    d_h = D // H
    Dkv = H_kv * d_h
    M_q  = N * L_q
    M_kv = N * L_k
    shapes: Dict[str, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = {
        "Q = X Wq": ((M_q, D),   (D, D),   (M_q, D)),
        "K = X Wk": ((M_kv, D),  (D, Dkv), (M_kv, Dkv)),
        "V = X Wv": ((M_kv, D),  (D, Dkv), (M_kv, Dkv)),
    }
    print("GEMM shapes:", shapes)


def main() -> None:
    spec = LlamaSpec(
        num_hidden_layers=2,
        hidden_size=768,
        num_attention_heads=24,
        num_key_value_heads=8,   # G = 3
        intermediate_size=1536,
        vocab_size=1000,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    N, L_q, L_k = 2, 16, 24
    D, H, H_kv = spec.hidden_size, spec.num_attention_heads, spec.num_key_value_heads

    print_projection_shapes(N, L_q, L_k, D, H, H_kv)

    x_q  = torch.randn(N, L_q, D)
    x_kv = torch.randn(N, L_k, D)

    # Expanded (baseline)
    attn_exp = MultiHeadAttention(spec)
    y_exp = attn_exp(values=x_kv, keys=x_kv, query=x_q)
    print("expanded out:", tuple(y_exp.shape))

    # Grouped (true GQA)
    attn_grp = MultiHeadGroupedAttention(spec)
    with torch.no_grad():
        attn_grp.load_state_dict(attn_exp.state_dict(), strict=True)  # sync weights
    y_grp = attn_grp(values=x_kv, keys=x_kv, query=x_q)
    print("grouped out:", tuple(y_grp.shape))
    print("max |expanded - grouped| =", (y_exp - y_grp).abs().max().item())

    # FlashAttention (expanded for SDPA)
    attn_flash = MultiHeadFlashAttention(spec)
    with torch.no_grad():
        attn_flash.load_state_dict(attn_exp.state_dict(), strict=True)
    y_flash = attn_flash(values=x_kv, keys=x_kv, query=x_q)
    print("flash out:", tuple(y_flash.shape))

    # FlexAttention (grouped-aware if available)
    attn_flex = MultiHeadFlexAttention(spec)
    with torch.no_grad():
        attn_flex.load_state_dict(attn_exp.state_dict(), strict=True)
    y_flex = attn_flex(values=x_kv, keys=x_kv, query=x_q)
    print("flex out:", tuple(y_flex.shape))

    # PagedAttention (decode-like usage)
    attn_paged = MultiHeadPagedAttention(spec, page_size=8)
    with torch.no_grad():
        # prefill: write all keys/values once
        y_prefill = attn_paged(values=x_kv, keys=x_kv, query=x_q)
        # subsequent decode step: no new kv provided, attend to cached pages
        y_decode = attn_paged(values=None, keys=None, query=x_q)
    print("paged prefill out:", tuple(y_prefill.shape))
    print("paged decode  out:", tuple(y_decode.shape))

    # Latent bottleneck attention
    attn_lat = MultiHeadLatentAttention(spec, latent_size=8)
    y_lat = attn_lat(values=x_kv, keys=x_kv, query=x_q)
    print("latent out:", tuple(y_lat.shape))


if __name__ == "__main__":
    main()
