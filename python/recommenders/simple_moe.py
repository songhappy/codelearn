import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # 使用 softmax 将输出转换为概率分布
        return F.softmax(self.fc(x), dim=1)

class MOE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts):
        super(MOE, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        # 计算每个专家的权重
        weights = self.gating_network(x)
        
        # 计算每个专家的输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        # 加权组合专家的输出
        output = torch.sum(weights.unsqueeze(-1) * expert_outputs, dim=1)
        return output
    
class YouTubeRecommendationSystem(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super(YouTubeRecommendationSystem, self).__init__()
        self.moe = MOE(input_dim, hidden_dim, 1, num_experts)  # 输出维度为 1（点击概率）

    def forward(self, x):
        # 使用 MOE 模型预测点击概率
        return torch.sigmoid(self.moe(x))
    
# 假设输入维度为 100，隐藏层维度为 64，专家数量为 4
input_dim = 100
hidden_dim = 64
num_experts = 4
model = YouTubeRecommendationSystem(input_dim, hidden_dim, num_experts)

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 假设我们有 1000 个样本，每个样本的输入维度为 100
num_samples = 1000
x = torch.randn(num_samples, input_dim)  # 输入数据
y = torch.randint(0, 2, (num_samples, 1)).float()  # 标签（0 或 1）

# 训练过程
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    outputs = model(x)
    loss = criterion(outputs, y)
    
    # 反向传播和优化
    loss.backward()
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    test_x = torch.randn(10, input_dim)  # 测试数据
    predictions = model(test_x)
    print("Predictions:", predictions)
