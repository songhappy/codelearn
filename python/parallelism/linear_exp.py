import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ----- Define Toy Model -----
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1024, 128)  # like ColwiseParallel
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 256)   # like RowwiseParallel

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# ----- Create Fake Dataset -----
def create_dataset(num_samples=1000):
    x = torch.randn(num_samples, 1024)       # input features
    y = torch.randint(0, 256, (num_samples,))  # simulate classification (0–255)
    return TensorDataset(x, y)

# ----- Train Function -----
def train_model(model, dataloader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# ----- Main -----
if __name__ == "__main__":
    dataset = create_dataset(1000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ToyModel()
    print("Model structure:")
    print(model)    
    print(model.linear1.weight.shape, model.linear2.weight.shape)
    train_model(model, dataloader)
