# https://pytorch.org/tutorials/beginner/basics/intro.html
#  || DataTypes || Tensors || Datasets & DataLoaders || Build Model || Autograd || Optimization || Save & Load Model || Multi-GPU Training and serving 

# --------------------------------
# Chapter 1: data type
# --------------------------------
# Summary Table:
# Data Type	PyTorch Alias	Description
# Floating-Point Types		
# torch.float32	torch.float	32-bit floating-point (single precision) 3.4×10^38 - 3.4×10^38
# torch.float64	torch.double	64-bit floating-point (double precision)
# torch.float16	torch.half	16-bit floating-point (half precision)   −6.55×10^4 - 6.55×10^4
 
# Integer Types		
# torch.int8		8-bit signed integer
# torch.uint8		8-bit unsigned integer
# torch.int16	torch.short	16-bit signed integer
# torch.int32	torch.int	32-bit signed integer
# torch.int64	torch.long	64-bit signed integer (default integer)
# Boolean Type		
# torch.bool		Boolean (True/False)
# Complex Types		
# torch.complex64		64-bit complex number (32-bit real + imag)
# torch.complex128		128-bit complex number (64-bit real + imag)
# Quantized Types		
# torch.quint8		8-bit unsigned quantized integer
# torch.qint8		8-bit signed quantized integer
# torch.qint32		32-bit signed quantized integer

# Key Differences Between float16 and bfloat16
# Format	Exponent Bits	Mantissa Bits	Dynamic Range	Precision (mantissa bits)
# float16	5	10	Narrower range, lower memory	Higher precision than bfloat16
# bfloat16	8	7	Same range as float32, faster	Lower precision than float16
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)  # same range as float32, faster
print(x)

# --------------------------------
# Chapter 2: torch tensor
# -------------------------------- 
import torch
import numpy as np

# 1. Creation Operations
# Creating tensors:
import torch

data =[1, 2, 3, 4]
data_tensor = torch.tensor([[1, 2], [3, 4]]) # Creates a tensor from a list
empty_tensor = torch.empty(2, 3) # Creates an uninitialized tensor
zeros_tensor = torch.zeros(2, 3) # Creates a tensor filled with zeros
rand_tensor = torch.rand(2, 3) # Creates a tensor with random values from a uniform distribution
ones_tensor = torch.ones(2, 3) # Creates a tensor filled with ones
range_tensor = torch.arange(0, 10, 2) # Creates a tensor with a range of values
linspace_tensor = torch.linspace(0, 1, steps=5) # Creates a tensor with evenly spaced values
np_array = np.array(data)
x_tensor = torch.from_numpy(np_array) #from a numpy array
x_np = x_tensor.numpy()
x_ones = torch.ones_like(data_tensor) # retains the properties of x_data # from another tensor
x_rand = torch.rand_like(data_tensor, dtype=torch.float) # overrides the datatype of x_data

# 2. Indexing and Slicing
# Access an element
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
element = tensor[1, 2]  # Access the element in the 2nd row, 3rd column
column_slice = tensor[:, 1]  # Accessing all rows for a specific column
row_slice = tensor[0, :]  # # Accessing a slice

# 3. Manipulation Operations
# Reshaping and Resizing

tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

reshaped_tensor = tensor.view(3, 2)  # Reshapes to 3 rows and 2 columns
reshape_tensor = tensor.reshape(3, 2) # Similar to view, but may return a copy if necessary, not contiguous, less efficient, more flexible

transposed_tensor = tensor.t() # Transpose the tensor
permute_tensor = tensor.permute(1, 0)  # Permute dimensions#  Swap rows and columns

# Concatenation and Splitting
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])
 
# Concatenate tensors along an existing dimention, here the first dimension (rows)
cat_tensor = torch.cat((tensor1, tensor2), dim=0)
torch.concatenate((tensor1, tensor2), dim=1)  # Concatenate along the second dimension (columns) same as torch.cat, shape of cat_tensor will be (4, 2)
stack_tensor = torch.stack((tensor1, tensor2), dim=0) # Stack tensors along a new dimension, shape of stack_tensor will be (2, 2, 2)

split_tensor = torch.split(tensor1, 1, dim=0)  # # Split tensor into chunks, here Split into chunks of size 1
print("Split Tensors:")
for t in split_tensor:
    print(t)

# Squeezing and Unsqueezing

tensor = torch.tensor([[1], [2], [3]])

squeezed_tensor = tensor.squeeze() # Remove dimensions of size 1
unsqueezed_tensor = squeezed_tensor.unsqueeze(0) # Add a dimension of size 1 at the specified position

#4. Arithmetic Operations
tensor1 = torch.tensor([[1., 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])


add_result = tensor1 + tensor2 # Element-wise addition
sub_result = tensor1 - tensor2 # Element-wise subtraction
mul_result = tensor1 * tensor2 # Element-wise multiplication
div_result = tensor1 / tensor2 # Element-wise division

# Using torch.add
add_result_torch = torch.add(tensor1, tensor2) # Element-wise addition
sub_result_torch = torch.sub(tensor1, tensor2) # Element-wise subtraction
mul_result_torch = torch.mul(tensor1, tensor2) # Element-wise multiplication
div_result_torch = torch.div(tensor1, tensor2) # Element-wise division


# In-place operations
tensor1.add_(tensor2)  # In-place addition
tensor1.sub_(tensor2)  # In-place subtraction
tensor1.mul_(tensor2)  # In-place multiplication
tensor1.div_(tensor2)  # In-place division
tensor1.add_(5)  # Add a scalar value to all elements
tensor1.sub_(5)  # Subtract a scalar value from all elements
tensor1.mul_(5)  # Multiply all elements by a scalar value   
tensor1.div_(5)  # Divide all elements by a scalar value

# 5. Reduction Operations

tensor = torch.tensor([[1., 2, 3], [4, 5, 6]])

total_sum = tensor.sum()# Sum of all elements
mean_columns = tensor.mean(dim=0) # Mean along the first dimension (column-wise)
max_value, max_index = tensor.max(dim=1)# Maximum value and its index along the second dimension (row-wise)
min_value, min_index = tensor.min(dim=0)# Minimum value and its index along the first dimension (column-wise)
prod_result = tensor.prod(dim=1) # Product of elements along a specified dimension

# 6. Comparison Operations
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[3, 2], [1, 0]])
# Element-wise comparison
comparison_gt = tensor1 > tensor2 # Element-wise greater than
comparison_eq = tensor1 == tensor2 # Element-wise equality

# 7. Matrix Operations
matrix1 = torch.tensor([[1., 2], [3, 4]])
matrix2 = torch.tensor([[5., 6], [7, 8]])

matrix_product = torch.mm(matrix1, matrix2) # Matrix multiplication, Requires two input tensors of shape (m, n) and (n, p), where m, n, and p are positive integers.

batch_matrix1 = torch.randn(10, 2, 3)  # Batch of 10 matrices, each 2x3
batch_matrix2 = torch.randn(10, 3, 4)  # Batch of 10 matrices, each 3x4
batch_product = torch.bmm(batch_matrix1, batch_matrix2) # Batch matrix multiplication, loop over the batch dimension
print("Batch Matrix Product Shape:", batch_product.shape)
matmul_result = torch.matmul(matrix1, matrix2) # General matrix multiplication (broadcasts if needed)
# torch.matmul: Can accept various shapes:
# 1D tensors: Treated as row or column vectors. # 2D tensors: Treated as matrices. # N-D tensors: Multiplies the last two dimensions and handles the rest accordingly.

inverse_matrix = torch.inverse(matrix1.float())  # # Compute the inverse of a square matrix, Ensure float for inversion
det_matrix = torch.det(matrix1.float())# Compute the determinant of a square matrix
eig_values, eig_vectors = torch.linalg.eig(matrix1.float()) # Eigenvalues and eigenvectors

# 8. Element-wise Functions
tensor = torch.tensor([[1, 2], [3, 4]])
exp_tensor = torch.exp(tensor) # Element-wise exponential
log_tensor = torch.log(tensor.float())  # Element-wise logarithm # Convert to float to apply log
abs_tensor = torch.abs(tensor.float())  # Element-wise absolute value
neg_tensor = torch.neg(tensor.float())  # Element-wise negation
square_tensor = torch.square(tensor.float())  # Element-wise square
sqrt_tensor = torch.sqrt(tensor.float())  # Element-wise square root
sin_tensor = torch.sin(tensor.float())  # Element-wise sine
cos_tensor = torch.cos(tensor.float())  # Element-wise cosine


# 9. Broadcasting Operations
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([10])  # 1D tensor
# Broadcasting PyTorch will automatically broadcast tensor2 to match the shape of tensor1.
broadcasted_sum = tensor1 + tensor2  #  the 1D tensor tensor2 is effectively treated as if it were reshaped to (1, 1) to align with the shape of tensor1 during the addition. The result will be [[11, 12], [13, 14]]

#10. Type Conversion
float_tensor = torch.tensor([[1.5, 2.5], [3.5, 4.5]])

int_tensor = float_tensor.int() # Converting to integer
long_tensor = float_tensor.long() # Converting to long
byte_tensor = float_tensor.byte() # Converting to byte
bool_tensor = float_tensor.bool() # Converting to boolean


# --------------------------------
# chapter 3: datasets and dataloader
# --------------------------------
# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]  # Returns a tuple (data, label) MUST BE TUPLE

# Example data
data = torch.tensor([[1, 2], [3, 4], [5, 6]])
labels = torch.tensor([0, 1, 0])

# Create a custom dataset
custom_dataset = CustomDataset(data, labels)
tensor_dataset = TensorDataset(data, labels)

# Create a DataLoader
dataloader = DataLoader(custom_dataset, batch_size=2, shuffle=True)

for batch in dataloader:
    print(batch)

# more about torch dataset and dataloader: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# and data_loader.py in this repo

# --------------------------------
# chapter 4: build model
# --------------------------------
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

from torch import nn 
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
print(model)
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# --------------------------------
# chapter 5: Automatic Differentiation
# --------------------------------

# 1. Tensor Gradients 
# https://blog.csdn.net/zandaoguang/article/details/108860486?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-108860486-blog-114977141.235%5Ev43%5Epc_blog_bottom_relevance_base4&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-108860486-blog-114977141.235%5Ev43%5Epc_blog_bottom_relevance_base4&utm_relevant_index=6
x = np.array([1.0, 2.0, 3.0])
x = torch.tensor(x)
print(x.requires_grad)

# 

# --------------------------------
# chapter 6: Optimization
# --------------------------------
# 1. Loss Functions
# Common loss functions include nn.MSELoss (Mean Square Error) for regression tasks, and nn.NLLLoss (Negative Log Likelihood) for classification. 
# nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss.
# CrossEntropyLoss is combined with raw logits, which allows more numerical stability during training., while NLLLoss expects log probabilities.
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

# 2. Optimizers
# Optimizers are used to update the weights of the model during training.
# Common optimizers include torch.optim.SGD (Stochastic Gradient Descent), torch.optim.Adam, and torch.optim.RMSprop.
# https://pytorch.org/docs/stable/optim.html

# 3. Training Loop
# The training loop consists of iterating over the dataset, making predictions, calculating the loss, computing gradients, and updating the weights.

# 4. Validation Loop
# 5. Testing Loop
# 6. Saving and Loading Models
# save and load the model parameters
import torchvision.models as models
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')  #store the learned parameters in an internal state dictionary, called state_dict
model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()
#. Save the entire model
torch.save(model, 'model.pth')
model = torch.load('model.pth', weights_only=False),


# --------------------------------
# chapter 7: Multi-GPU Training and Serving
# --------------------------------
# 1. Data Parallelism
# Data parallelism is a technique to distribute the training data across multiple GPUs and perform parallel computations.
# PyTorch provides the nn.DataParallel module to handle data parallelism.
# https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html

# 2. Model Parallelism
# Model parallelism is a technique to distribute the model across multiple GPUs and perform parallel computations.
# PyTorch provides the torch.nn.parallel module to handle model parallelism.
# https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html
# see model_parallel.py in this repo

# 3. Distributed Training
# Distributed training is a technique to train a model across multiple devices, machines, or nodes.
# PyTorch provides the torch.nn.parallel.DistributedDataParallel module to handle distributed training.
# https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
# see distributed_training.py in this repo

# 4. Serving Models
# Serving models is the process of deploying a trained model for inference in production environments.
# PyTorch provides the torch.jit module to compile and optimize models for serving.
# https://pytorch.org/tutorials/advanced/cpp_export.html
# see serving_model.py in this repo