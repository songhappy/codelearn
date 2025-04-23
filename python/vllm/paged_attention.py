# Paged KV Caching: Instead of storing KV caches in a contiguous tensor, PagedAttention splits the cache into smaller blocks (pages), reducing memory fragmentation.
# Efficient Attention Computation: Queries attend only to the relevant pages in memory.
# High-throughput Inference: Optimized for LLM serving by enabling parallel processing of different sequences.
import torch

class PagedAttention:
    def __init__(self, d_model, num_heads, block_size=128, max_blocks=1024):
        """
        PagedAttention with block-wise KV caching.

        Args:
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            block_size (int): Size of each KV cache block.
            max_blocks (int): Maximum number of blocks per sequence.
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.d_head = d_model // num_heads

        # Initialize paged KV cache (block-wise storage)
        self.kv_cache = {
            "keys": torch.zeros((max_blocks, num_heads, block_size, self.d_head), dtype=torch.float16),
            "values": torch.zeros((max_blocks, num_heads, block_size, self.d_head), dtype=torch.float16),
            "block_indices": {}  # Store block allocations per batch
        }

    def allocate_block(self, batch_idx):
        """Allocates a new block for KV cache for a given batch index."""
        if batch_idx not in self.kv_cache["block_indices"]:
            self.kv_cache["block_indices"][batch_idx] = []
        
        if len(self.kv_cache["block_indices"][batch_idx]) >= self.max_blocks:
            raise RuntimeError(f"KV cache exceeded max blocks for batch {batch_idx}.")

        block_idx = len(self.kv_cache["block_indices"][batch_idx])
        self.kv_cache["block_indices"][batch_idx].append(block_idx)
        return block_idx

    def forward(self, q, k, v, attn_mask=None):
        """
        Compute attention with paged KV caching.
        
        Args:
            q (Tensor): Query tensor of shape (batch_size, seq_len, d_model).
            k (Tensor): Key tensor of shape (batch_size, seq_len, d_model).
            v (Tensor): Value tensor of shape (batch_size, seq_len, d_model).
            attn_mask (Tensor, optional): Attention mask.

        Returns:
            Tensor: Output of attention mechanism.
        """
        batch_size, seq_len, _ = q.shape

        # Reshape Q, K, V for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.d_head).permute(0, 2, 1, 3)  # [B, H, S, D]
        k = k.view(batch_size, seq_len, self.num_heads, self.d_head).permute(0, 2, 1, 3)  # [B, H, S, D]
        v = v.view(batch_size, seq_len, self.num_heads, self.d_head).permute(0, 2, 1, 3)  # [B, H, S, D]

        # Allocate KV cache blocks for each batch
        for b in range(batch_size):
            if b not in self.kv_cache["block_indices"]:
                block_idx = self.allocate_block(b)
            else:
                block_idx = self.kv_cache["block_indices"][b][-1]

            # Store KV cache per batch in allocated block
            self.kv_cache["keys"][block_idx, :, :seq_len, :] = k[b]
            self.kv_cache["values"][block_idx, :, :seq_len, :] = v[b]

        # Compute attention scores (QK^T / sqrt(d))
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)  # [B, H, S, S]

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)

        # Compute attention output
        output = torch.matmul(attn_probs, v)  # [B, H, S, D]
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)  # Reshape back

        return output

# Example usage
d_model = 512
num_heads = 8
seq_len = 128
batch_size = 2  # Now testing with batch size > 1

paged_attn = PagedAttention(d_model, num_heads)

# Simulated input tensors
q = torch.randn(batch_size, seq_len, d_model)
k = torch.randn(batch_size, seq_len, d_model)
v = torch.randn(batch_size, seq_len, d_model)

output = paged_attn.forward(q, k, v)
print("PagedAttention Output Shape:", output.shape)  # Expected: [2, 128, 512]

# Print allocated blocks
print("Allocated Blocks:", paged_attn.kv_cache["block_indices"])
