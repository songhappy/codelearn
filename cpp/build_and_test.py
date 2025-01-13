from torch.utils.cpp_extension import load

# Compile and load the C++ extension
custom_ops = load(name="custom_ops", sources=["src/custom_ops.cpp"])

# Test the custom addition function
import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Call the custom C++ function
result = custom_ops.custom_add(a, b)
print(f"Result of custom_add: {result}")
