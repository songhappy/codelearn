import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# RMSNorm (Layer Normalization Alternative)
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight

# Rotary Position Embedding (RoPE)
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim  # Should be head_dim, not full hidden_size
        self.max_seq_len = max_seq_len

        # Compute inverse frequency for RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device=None):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        # Einsum allows computing many common multi-dimensional linear algebraic array operations by representing them
        # in a short-hand format based on the Einstein summation convention,
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, head_dim // 2)

        # Correctly shape cos and sin
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)
        return emb.cos().to(device), emb.sin().to(device)

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def apply_rope(self, x, cos, sin):
        """
        Applies RoPE (Rotary Position Embedding) to the given tensor.
        """
        x_1, x_2 = x[..., 0::2], x[..., 1::2]  # Split into even and odd parts

        # Ensure cos/sin match (batch_size, num_heads, seq_len, head_dim / 2)
        cos = cos[..., : x_1.shape[-1]]
        sin = sin[..., : x_1.shape[-1]]

        x_rot = x_1 * cos - x_2 * sin
        x_imag = x_1 * sin + x_2 * cos

        return torch.cat([x_rot, x_imag], dim=-1)  # Recombine into original shape


    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Get RoPE cos/sin embeddings
        cos, sin = self.rotary_emb(seq_len, x.device)

        # Reshape for proper broadcasting
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)

        # Apply RoPE correctly
        q = self.apply_rope(q, cos, sin)
        k = self.apply_rope(k, cos, sin)

        # Scale before dot product
        q = q * self.scale
        attn_weights = q @ k.transpose(-2, -1)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(attn_output)


# Mixture of Experts (MoE)
class MoE(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.experts = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)])
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        gates = F.softmax(self.gate(x), dim=-1)  # Compute gating scores (batch_size, seq_length, num_expoerts)
        # top_k_gates(batch_size, seq_length, top_k), top_k_idx(s(batch_size, seq_length, top_k)
        top_k_gates, top_k_idx = torch.topk(gates, self.top_k, dim=-1)  # Get top-k expert indices

        # Correct indexing to avoid "only integer tensors can be converted to an index" error
        batch_size, seq_length, _ = x.shape  #x(batch_size, seq_length, hidden_dim)
        expert_outputs = torch.zeros_like(x)  # Initialize output

        for i in range(self.top_k):
            idx = top_k_idx[..., i]  # Selects the i-th expert index for all batches and sequence positions.
            for b in range(batch_size):
                for s in range(seq_length):
                    expert_outputs[b, s] += top_k_gates[b, s, i] * self.experts[idx[b, s].item()](x[b, s])

        return expert_outputs


# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num_experts):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads)
        self.norm2 = RMSNorm(hidden_size)
        self.moe = MoE(hidden_size, num_experts)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.moe(self.norm2(x))
        return x

# Full DeepSeek V3 Model
class DeepSeekV3(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, num_experts):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(hidden_size, num_heads, num_experts) for _ in range(num_layers)])
        self.norm = RMSNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, mask=None):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.output(x)



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model parameters
vocab_size = 100  # Small vocabulary for demo
hidden_size = 128
num_layers = 2
num_heads = 8
num_experts = 6
seq_length = 10
batch_size = 4
num_epochs = 5

# Initialize model
model = DeepSeekV3(vocab_size, hidden_size, num_layers, num_heads, num_experts).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Generate dummy data (batch of tokenized sequences)
input_data = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
target_data = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Forward pass
    output = model(input_data)  # (batch_size, seq_length, vocab_size)
    
    # Reshape for loss calculation
    loss = criterion(output.view(-1, vocab_size), target_data.view(-1))
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
