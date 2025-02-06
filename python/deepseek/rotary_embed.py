import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Create position tensor
        position = torch.arange(max_seq_len).unsqueeze(1)  # Shape: (max_seq_len, 1)

        # Create frequency term (inverse log scale)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))

        # Compute sinusoidal signals
        pos_enc = torch.zeros(max_seq_len, dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pos_enc[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices

        # Register buffer (non-trainable)
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, x):
        """
        Adds positional encoding to the input tensor.
        x: (batch_size, seq_len, dim)
        """
        return x + self.pos_enc[:x.shape[1], :]


class RotaryEmbedding(nn.Module): 
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim  # Should be head_dim, not full hidden_size

        # Compute inverse frequency
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device=None):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, head_dim // 2)

        # Apply sin and cos
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)
        return emb.cos().to(device), emb.sin().to(device)



seq_len = 10
dim = 16  # Small demo dimension

# Initialize embeddings
sinusoidal = SinusoidalPositionalEncoding(dim)
rope = RotaryEmbedding(dim)

# Generate embeddings
x = torch.randn(1, seq_len, dim)  # Dummy input
sin_embedded = sinusoidal(x)[0].detach().numpy()  # Extract first batch
cos, sin = rope(seq_len)
rope_embedded = cos.numpy()  # Only showing cos for visualization

# Plot
plt.ion()
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(sin_embedded, aspect='auto', cmap='coolwarm')
plt.title("Sinusoidal Positional Encoding")

plt.subplot(1, 2, 2)
plt.imshow(rope_embedded, aspect='auto', cmap='coolwarm')
plt.title("Rotary Positional Embedding (RoPE)")
matplotlib.use('Agg') 

plt.show()
