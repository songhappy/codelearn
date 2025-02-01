import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Step 1: Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Step 2: Apply Standard Scaler
standard_scaler = StandardScaler()
X_standard_scaled = standard_scaler.fit_transform(X)

# Step 3: Apply Min-Max Scaler (for demonstration, not necessary to use both)
minmax_scaler = MinMaxScaler()
X_minmax_scaled = minmax_scaler.fit_transform(X)

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_standard_scaled, y, test_size=0.2, random_state=42)

# Step 5: Create DataLoader
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 6: Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(20, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

model = SimpleModel()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 7: Training loop
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    for features, labels in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(features).squeeze()  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# Step 8: Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for features, labels in test_loader:
        outputs = model(features).squeeze()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy: {correct / total * 100:.2f}%')

# do prediction
model.eval()
with torch.no_grad():
    X_new = torch.tensor(X_test[:10], dtype=torch.float32)
    y_pred = model(X_new)
    print(y_pred)
    print((y_pred > 0.5).float())
    print(y_test[:10])

