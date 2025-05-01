import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import optim
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed._composable.fsdp import CPUOffloadPolicy, fully_shard

# ------------------ Memory Hook Tools ------------------

def get_memo():
    device = torch.device(f"xpu:{torch.xpu.current_device()}")
    current = torch.xpu.memory_allocated(device)
    peak_active = torch.xpu.memory_stats(device)["active_bytes.all.peak"]
    peak_alloc = torch.xpu.max_memory_allocated(device)
    peak_reserved = torch.xpu.max_memory_reserved(device)
    return current, peak_active, peak_alloc, peak_reserved

def forward_hook(module, input, output):
    curr, peak_act, peak_alloc, peak_res = get_memo()
    print(f"\n[FWD HOOK] {module.__class__.__name__} | Memory(B): {curr}, {peak_act}, {peak_alloc}, {peak_res}")

def backward_hook(module, grad_input, grad_output):
    curr, peak_act, peak_alloc, peak_res = get_memo()
    print(f"\n[BWD HOOK] {module.__class__.__name__} | Memory(B): {curr}, {peak_act}, {peak_alloc}, {peak_res}")


# ------------------ Model Components ------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        assert self.head_dim * num_heads == embed_size
        self.v_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.k_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        values = self.v_proj(values)
        keys = self.k_proj(keys)
        queries = self.queries(query)
        values = values.reshape(N, -1, self.num_heads, self.head_dim)
        keys = keys.reshape(N, -1, self.num_heads, self.head_dim)
        queries = queries.reshape(N, -1, self.num_heads, self.head_dim)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.embed_size ** 0.5), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.reshape(N, -1, self.num_heads * self.head_dim)
        return self.fc_out(out)

class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, embed_size)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_size, dropout):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, ff_hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention = self.attention(x, x, x, mask)
        x = self.norm1(attention + x)
        forward = self.feed_forward(x)
        return self.norm2(forward + x)

class SimplifiedLLaMAModel(nn.Module):
    def __init__(self, embed_size, num_layers, heads, ff_hidden_size, vocab_size, max_length, dropout):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, ff_hidden_size, dropout) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, mask)
        return self.fc_out(out)

class TokenDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

# ------------------ Training ------------------

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    i = 0
    for batch, (input_data, target_data) in enumerate(dataloader):
        i = i + 1
        input_data, target_data = input_data.to(device), target_data.to(device)
        optimizer.zero_grad()
        output = model(input_data)
        output = output.view(-1, output.shape[-1])
        target_data = target_data.view(-1)
        loss = criterion(output, target_data)
        loss.backward()
        optimizer.step()
        torch.xpu.synchronize()
        print(f"[Rank {dist.get_rank()}] Batch {batch}, {i}, Loss: {loss.item()}")
    return

def setup_fsdp_training():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("xccl", rank=rank, world_size=world_size)
    torch.xpu.set_device(rank)
    device = torch.device(f"xpu:{rank}")

    # Model setup
    model = SimplifiedLLaMAModel(
        embed_size=2048,
        num_layers=10,
        heads=8,
        ff_hidden_size=2048,
        vocab_size=100000,
        max_length=100,
        dropout=0.1
    ).to(device)

    # # Register memory hooks
    # for name, module in model.named_modules():
    #     if len(list(module.children())) == 0:
    #         module.register_forward_hook(forward_hook)
    #         module.register_full_backward_hook(backward_hook)

    num_layers_sharded = 0
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear):
            fully_shard(module)
            num_layers_sharded += 1

    if num_layers_sharded == 0:
        raise ValueError(
            "No layer modules were sharded. Please check if shard conditions are working as expected."
        )

    # Finally shard the entire model to account for any stragglers
    fully_shard(model)

    # Data setup
    data = torch.randint(0, 100000, (64, 100))
    targets = torch.cat([data[:, 1:], torch.zeros(64, 1, dtype=torch.long)], dim=1)
    dataset = TokenDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=8)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(2):
        print(f"\n[Rank {rank}] Epoch {epoch}")
        train(model, dataloader, optimizer, criterion, device)

    dist.destroy_process_group()

if __name__ == "__main__":
    setup_fsdp_training()
