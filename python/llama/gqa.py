import torch
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_query_heads, num_kv_heads):
        super().__init__()
        assert num_query_heads % num_kv_heads == 0, "Query heads must be divisible by KV heads"

        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_query_heads
        self.kv_group_size = num_query_heads // num_kv_heads  # How many Q heads share a KV head

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)  
        self.W_k = nn.Linear(embed_dim, embed_dim // self.kv_group_size, bias=False)  
        self.W_v = nn.Linear(embed_dim, embed_dim // self.kv_group_size, bias=False)  
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_query_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Expand K and V to match query heads
        K = K.unsqueeze(2).expand(-1, -1, self.kv_group_size, -1, -1).reshape(batch_size, seq_len, self.num_query_heads, self.head_dim)
        V = V.unsqueeze(2).expand(-1, -1, self.kv_group_size, -1, -1).reshape(batch_size, seq_len, self.num_query_heads, self.head_dim)

        # Compute scaled dot-product attention
        attn_scores = torch.einsum("bqhd,bkhd->bhqk", Q, K) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Apply attention weights to values
        attn_output = torch.einsum("bhqk,bqhd->bqhd", attn_weights, V)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        return self.out_proj(attn_output)

# Example usage
embed_dim = 4096  # LLaMA 3 model dimension
num_query_heads = 32
num_kv_heads = 8  # 4 query heads share 1 KV head

gqa = GroupedQueryAttention(embed_dim, num_query_heads, num_kv_heads)
x = torch.rand(1, 128, embed_dim)  # Batch of 1, 128 tokens, 4096 dim
output = gqa(x)
print("Output shape:", output.shape)  # Expected: torch.Size([1, 128, 4096])
