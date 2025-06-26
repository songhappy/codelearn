# ----------------------------
# File: train_single.py
# ----------------------------

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from myutils import get_device_bkd, get_xpu_memory_used_from_xpu_smi, get_gpu_memory_used_from_nvidia_smi

# ----------------------------
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(1000, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train():

    device_type, backend, torch_device= get_device_bkd()
    model = SimpleModel().to(device_type)

    X = torch.randn(100, 1000)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(1):
        for step, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device_type)
            y_batch = y_batch.to(device_type)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            memory_log = (
                get_xpu_memory_used_from_xpu_smi(f"Epoch {epoch}, Step {step}", )
                if device_type == "xpu"
                else get_gpu_memory_used_from_nvidia_smi(f"Epoch {epoch}, Step {step}",)
            )
            print(f"[Single Device] Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}, {memory_log}")

if __name__ == "__main__":
    print("Running in single-device mode")
    train()
