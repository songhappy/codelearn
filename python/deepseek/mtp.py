import torch
import torch.nn as nn
import torch.optim as optim

class TransformerMTP(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, hidden_dim, num_tokens_pred):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, num_heads, num_layers, dim_feedforward=hidden_dim)
        self.output_heads = nn.Linear(d_model, vocab_size * num_tokens_pred)  # MTP head

        self.num_tokens_pred = num_tokens_pred
        self.vocab_size = vocab_size

    def forward(self, x, tgt_mask=None):
        """
        x: (batch_size, seq_len)
        Returns: (batch_size, seq_len, num_tokens_pred, vocab_size)
        """
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = self.transformer(x, x, tgt_mask=tgt_mask)  # Apply transformer
        logits = self.output_heads(x)  # (batch_size, seq_len, num_tokens_pred * vocab_size)
        
        # Reshape output to predict multiple tokens per position
        logits = logits.view(x.shape[0], x.shape[1], self.num_tokens_pred, self.vocab_size)
        return logits

# Example Hyperparameters
vocab_size = 10000
d_model = 512
num_layers = 6
num_heads = 8
hidden_dim = 2048
num_tokens_pred = 3  # Predict 3 tokens per position

# Create model
model = TransformerMTP(vocab_size, d_model, num_layers, num_heads, hidden_dim, num_tokens_pred)

# Example input (batch_size=2, seq_len=5)
input_ids = torch.randint(0, vocab_size, (2, 5))  # Random tokenized input

# Forward pass
logits = model(input_ids)  # (batch_size, seq_len, num_tokens_pred, vocab_size)
print("Output logits shape:", logits.shape)  # Expected: (2, 5, 3, 10000)
