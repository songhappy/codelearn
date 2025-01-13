import torch
import torchvision.models as models

# Initialize ResNet-50 model
model = models.resnet50(pretrained=True)

# Set model to evaluation mode
model.eval()

# Use TorchDynamo with Inductor optimization
torch.compile(model, backend="inductor")


# Create a random input tensor
x = torch.randn(1, 3, 224, 224)

# Run the compiled model with Inductor optimizations
with torch.no_grad():
    output_inductor = model(x)

print(output_inductor)
