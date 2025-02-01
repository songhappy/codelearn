import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# DNN Expert
class DNNExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DNNExpert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# CNN Expert (for sequential data like video frames)
class CNNExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CNNExpert, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(2)  # Add channel dimension
        x = F.relu(self.conv1(x))
        x = x.mean(dim=2)  # Global average pooling
        x = self.fc(x)
        return x

# RNN Expert (for sequential data like watch history)
class RNNExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNExpert, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension
        _, h_n = self.rnn(x)
        x = self.fc(h_n.squeeze(0))
        return x

class SparseGatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, top_k):
        super(SparseGatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        logits = self.fc(x)
        top_k_weights, top_k_indices = torch.topk(F.softmax(logits, dim=1), self.top_k)
        return top_k_weights, top_k_indices
    
class MOE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, top_k):
        super(MOE, self).__init__()
        self.experts = nn.ModuleList([
            DNNExpert(input_dim, hidden_dim, output_dim),
            CNNExpert(input_dim, hidden_dim, output_dim),
            RNNExpert(input_dim, hidden_dim, output_dim),
            DNNExpert(input_dim, hidden_dim, output_dim),  # Default expert for cold start
        ])
        self.gating_network = SparseGatingNetwork(input_dim, num_experts, top_k)

    def forward(self, x):
        # Get top-k weights and indices from the gating network
        top_k_weights, top_k_indices = self.gating_network(x)
        
        # Get outputs from the top-k experts
        expert_outputs = torch.stack([self.experts[i](x) for i in top_k_indices[0]], dim=1)
        
        # Combine expert outputs using the top-k weights
        output = torch.sum(top_k_weights.unsqueeze(-1) * expert_outputs, dim=1)
        return output
    
class YouTubeRecommender(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, top_k):
        super(YouTubeRecommender, self).__init__()
        self.moe = MOE(input_dim, hidden_dim, 1, num_experts, top_k)  # Output is a single probability

    def forward(self, x):
        # Use sigmoid to convert the output to a probability
        return torch.sigmoid(self.moe(x))

def handle_cold_start(model, x, is_new_user=False, is_new_item=False):
    if is_new_user or is_new_item:
        # Use the default expert (last expert in the list)
        default_expert = model.moe.experts[-1]
        return torch.sigmoid(default_expert(x))
    else:
        return model(x)

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def train_distributed(model, train_loader, criterion, optimizer, local_rank):
    model = DDP(model.to(local_rank), device_ids=[local_rank])
    model.train()
    
    for epoch in range(num_epochs):
        for x, y in train_loader:
            x, y = x.to(local_rank), y.to(local_rank)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Hyperparameters
input_dim = 100  # Number of input features
hidden_dim = 64  # Hidden layer size for experts
num_experts = 4  # Number of experts
top_k = 2  # Number of experts to activate
num_samples = 1000  # Number of training samples
num_epochs = 10

# Create synthetic data
x = torch.randn(num_samples, input_dim)  # Input features
y = torch.randint(0, 2, (num_samples, 1)).float()  # Binary labels (0 or 1)

# Initialize the model, loss function, and optimizer
model = YouTubeRecommender(input_dim, hidden_dim, num_experts, top_k)
criterion = nn.BCELoss()  # Binary cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Distributed training setup
local_rank = setup_distributed()
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x, y), batch_size=32, shuffle=True
)

# Train the model
train_distributed(model, train_loader, criterion, optimizer, local_rank)
