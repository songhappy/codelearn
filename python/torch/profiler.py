import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler
from torch.utils.data import TensorDataset, DataLoader

# Example model: A simple neural network
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Generate some random data

batch_size = 4
inputs = torch.randn(1024, 784)  # Batch of 1024, 784 features (e.g., flattened MNIST images)
labels = torch.randint(0, 10, (1024,))  # Random labels for 10 classes
dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define model, loss, and optimizer
model = SimpleModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Use PyTorch Profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA  # Use CUDA if available
    ],
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_dir'),  # Save to log_dir for TensorBoard
    record_shapes=True,
    with_stack=True
) as profiler:
# if 1> 0:    
    # Example training loop
    model.train()
    for epoch in range(2):  # Train for 2 epochs
        for idx, batch in enumerate(dataloader):
            input_b, label_b = batch
            optimizer.zero_grad()  # Reset gradients
            output_b = model(input_b,)  # Forward pass
            loss = criterion(output_b, label_b)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            if idx % 10 == 0:
                print(f"Epoch [{epoch+1}], Step [{idx+1}], Loss: {loss.item():.4f}")
        # Record profiler data
        profiler.step()

# Print profiler results
print(profiler.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total", row_limit=10))

# tensorboard --logdir=./log_dir
