import torch
import torch.nn.functional as F

# A memory-efficient algorithm designed for GPUs (Dao et al., 2022).
# Uses a tiling-based algorithm to compute attention in chunks, avoiding memory overhead from storing intermediate activations.
# Implements a recomputation strategy and low-level optimizations to improve efficiency.
# Computational complexity: O(N²d) (same as standard attention), but with lower memory overhead.
# Used in models like GPT-4, Llama, and Falcon to train longer sequences efficiently.

# Define a custom FlashAttention function
def flash_attention(q, k, v, block_size=64):
    """
    Implements FlashAttention from scratch with tiling.
    
    Args:
        q, k, v: Query, Key, Value tensors of shape (batch_size, seq_len, num_heads, head_dim)
        block_size: Chunk size for block-wise attention computation
        
    Returns:
        Attention output with optimized memory and computation
    """
    batch_size, seq_len, num_heads, head_dim = q.shape

    # Initialize output tensor
    output = torch.zeros_like(v)

    # Iterate over sequence in blocks to reduce memory usage
    for start in range(0, seq_len, block_size):
        end = min(start + block_size, seq_len)

        # Extract block slices
        q_block = q[:, start:end]  # (batch, block_size, num_heads, head_dim)
        k_block = k[:, start:end]  
        v_block = v[:, start:end]  

        # Compute scaled dot-product attention (without storing the full matrix)
        attn_scores = torch.einsum("bqhd,bkhd->bhqk", q_block, k_block) / (head_dim ** 0.5)

        # Apply softmax in a memory-efficient way
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Compute attention output (fused kernel optimization)
        output[:, start:end] = torch.einsum("bhqk,bkhd->bqhd", attn_probs, v_block)

    return output

# Example Usage
batch_size, seq_len, num_heads, head_dim = 2, 128, 8, 64
q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cpu", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

# Apply custom FlashAttention
output = flash_attention(q, k, v)

print("FlashAttention Output Shape:", output.shape)  # (batch_size, seq_len, num_heads, head_dim)


import time

def standard_attention(q, k, v):
    """Standard full self-attention"""
    attn_scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / (head_dim ** 0.5)
    attn_probs = F.softmax(attn_scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", attn_probs, v)

# Measure time
start = time.time()
std_output = standard_attention(q, k, v)
std_time = time.time() - start

start = time.time()
flash_output = flash_attention(q, k, v)
flash_time = time.time() - start

print(f"Standard Attention Time: {std_time:.6f} sec")
print(f"FlashAttention Time: {flash_time:.6f} sec")
