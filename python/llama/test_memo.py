import torch 
import torch
import torch.nn as nn
import torchvision.models as models
import time 

torch_device = (
    torch.xpu if torch.xpu.is_available()
    else torch.cuda if torch.cuda.is_available()
    else torch.cpu
)

device = (
    'xpu' if torch.xpu.is_available()
    else 'cuda' if torch.cuda.is_available()
    else 'cpu'
)

_BYTES_IN_GIB = 1024 ** 3

def get_memo():

    peak_memory_active = (
        torch_device.memory_stats().get("active_bytes.all.peak", 0) / _BYTES_IN_GIB
    )
    peak_memory_alloc = torch_device.max_memory_allocated(device) / _BYTES_IN_GIB
    peak_memory_reserved = torch_device.max_memory_reserved(device) / _BYTES_IN_GIB
    return (time.time(), peak_memory_active, peak_memory_alloc, peak_memory_reserved)


# Initialize ResNet50
model = models.resnet50().to(device)
model.train()  # Set model to training mode

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

# Step 1: Define the Multi-Head Self-Attention mechanism
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # Define linear transformations for queries, keys, and values
        self.v_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.k_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)

        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]  # Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]


        # Apply the linear projections to get values, keys, and queries
        values = self.v_proj(values)  # Shape: (N, value_len, embed_size)
        keys = self.k_proj(keys)        # Shape: (N, key_len, embed_size)
        queries = self.queries(query) # Shape: (N, query_len, embed_size)

        # Split into multiple heads for multi-head attention
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.num_heads, self.head_dim)

        # Calculate the attention scores (energy)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N, heads, query_len, key_len)

        # Mask future tokens (for autoregressive models)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Apply the softmax to get attention scores
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)  # (N, heads, query_len, key_len)

        # Compute the output by applying attention to the values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])  # (N, query_len, heads, head_dim)

        # Concatenate heads and pass through the final linear layer
        out = out.reshape(N, query_len, self.num_heads * self.head_dim)
        out = self.fc_out(out)  # Shape: (N, query_len, embed_size)

        return out

# Step 2: Define the Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, embed_size)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

# Step 3: Define a Single Transformer Block (similar to LLaMA Block)
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_size, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = FeedForward(embed_size, ff_hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention = self.attention(x, x, x, mask)
        x = self.norm1(attention + x)  # Add & Norm 1
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)  # Add & Norm 2
        return out

# Step 4: Define the Transformer Model (LLaMA-like model)
class SimplifiedLLaMAModel(nn.Module):
    def __init__(self, embed_size, num_layers, heads, ff_hidden_size, vocab_size, max_length, dropout):
        super(SimplifiedLLaMAModel, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, ff_hidden_size, dropout) for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, mask)

        return self.fc_out(out)

# Step 5: Define a simple custom dataset class for tokenized data
class TokenDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data  # Input token sequences
        self.targets = targets  # Target token sequences (shifted right)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

# Step 6: Training function
def train(model, dataloader, optimizer, criterion, device):
    model.train()  # Set the model to training mode
    epoch_loss = 0
    # 🔹 Training Loop

    for batch, (input_data, target_data) in enumerate(dataloader):
        input_data = input_data.to(device)  # Move input data to GPU/CPU
        target_data = target_data.to(device)  # Move target data to GPU/CPU
        
        optimizer.zero_grad()  # Zero out gradients from previous batch

        print("before forward")
        now, peak_memory_active, peak_memory_alloc, peak_memory_reserved = get_memo()
        print( now, peak_memory_active, peak_memory_alloc, peak_memory_reserved)

        # Forward pass
        output = model(input_data)  # Shape: (batch_size, seq_len, vocab_size)
        
        print("after forward")
        now, peak_memory_active, peak_memory_alloc, peak_memory_reserved = get_memo()
        print( now, peak_memory_active, peak_memory_alloc, peak_memory_reserved)


        # Reshape output and target to match dimensions for cross-entropy loss
        output = output.view(-1, output.shape[-1])  # (batch_size * seq_len, vocab_size)
        target_data = target_data.view(-1)  # (batch_size * seq_len)

        # Calculate the loss (CrossEntropyLoss expects (N, C) and target (N))
        loss = criterion(output, target_data)

        # Backpropagation and optimization step
        loss.backward()
        print("after backward")
        now, peak_memory_active, peak_memory_alloc, peak_memory_reserved = get_memo()
        print( now, peak_memory_active, peak_memory_alloc, peak_memory_reserved)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

# Example usage with the simplified model
if __name__ == "__main__":
    vocab_size = 100000   # Number of tokens in the vocabulary
    embed_size = 2048     # Embedding size
    num_heads = 8            # Number of attention heads
    num_layers = 10       # Number of transformer layers
    ff_hidden_size = 2048 # Feedforward hidden size
    max_length = 100     # Maximum length of the input sequence
    dropout = 0.1        # Dropout rate
    max_generate_length = 20  # Maximum number of tokens to generate
    batch_size = 32      # Batch size for training
    num_epochs = 10000      # Number of epochs to train
    learning_rate = 1e-4 # Learning rate
    
    # Initialize the simplified LLaMA-like model
    model = SimplifiedLLaMAModel(embed_size, num_layers, num_heads, ff_hidden_size, vocab_size, max_length, dropout)
    
    print(model)
       # Example dummy data (replace with real tokenized data)
    # Assume the input sequences are tokenized sentences, and targets are shifted by one
    data = torch.randint(0, vocab_size, (1000, max_length))  # 1000 sequences of length max_length
    targets = torch.cat([data[:, 1:], torch.zeros(1000, 1, dtype=torch.long)], dim=1)  # Shifted target sequences


    # Create a dataset and dataloader
    dataset = TokenDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token (if any)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    # Training loop
    for epoch in range(num_epochs):

        epoch_loss = train(model, dataloader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')


