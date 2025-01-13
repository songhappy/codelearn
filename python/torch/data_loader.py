import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, default_collate
import math
import torchvision
from torch.utils.data import Dataset, DataLoader

# gradient computation etc. not efficient for whole data set
# -> divide dataset into small batches

'''
# training loop
for epoch in range(num_epochs):
    # loop over all batches
    for i in range(total_batches):
        batch_x, batch_y = ...
'''


# epoch = one forward and backward pass of ALL training samples
# batch_size = number of training samples used in one forward/backward pass
# number of iterations = number of passes, each pass (forward+backward) using [batch_size] number of sampes
# e.g : 100 samples, batch_size=20 -> 100/20=5 iterations for 1 epoch

# --> DataLoader can do the batch computation for us

# Implement a custom Dataset:
# inherit Dataset
# implement __init__ , __getitem__ , and __len__

class WineDataset(Dataset):

    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = np.loadtxt('/Users/guoqiong/intelWork/data/wine/wine.csv', delimiter=',',
                        dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(xy[:, 1:])  # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:, [0]])  # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


# # create dataset
# dataset = WineDataset()

# # get first sample and unpack
# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)

# # Load whole dataset with DataLoader
# # shuffle: shuffle data, good for training
# # num_workers: faster loading with multiple subprocesses
# # !!! IF YOU GET AN ERROR DURING LOADING, SET num_workers TO 0 !!!
# train_loader = DataLoader(dataset=dataset,
#                           batch_size=4,
#                           shuffle=True,
#                           num_workers=2)

# # convert to an iterator and look at one random sample
# dataiter = iter(train_loader)
# data = next(dataiter)
# features, labels = data
# print(features, labels)

# # Dummy Training loop
# num_epochs = 2
# total_samples = len(dataset)
# n_iterations = math.ceil(total_samples / 4)
# print(total_samples, n_iterations)
# for epoch in range(num_epochs):
#     for i, (inputs, labels) in enumerate(train_loader):

#         # here: 178 samples, batch_size = 4, n_iters=178/4=44.5 -> 45 iterations
#         # Run your training process
#         if (i + 1) % 5 == 0:
#             print(
#                 f'Epoch: {epoch + 1}/{num_epochs}, Step {i + 1}/{n_iterations}| Inputs {inputs.shape} | Labels {labels.shape}')

# # some famous datasets are available in torchvision.datasets
# # e.g. MNIST, Fashion-MNIST, CIFAR10, COCO

# train_dataset = torchvision.datasets.MNIST(root='/Users/guoqiong/intelWork/data/',
#                                            train=True,
#                                            transform=torchvision.transforms.ToTensor(),
#                                            download=True)

# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=3,
#                           shuffle=True)

# # look at one random sample
# dataiter = iter(train_loader)
# data = next(dataiter)
# inputs, targets = data
# print(inputs.shape, targets.shape)

# custom Dataset
import torch
from torch.utils.data import Dataset, DataLoader

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Example data
data = torch.tensor([[1, 2], [3, 4], [5, 6]])
labels = torch.tensor([0, 1, 0])

# Create a custom dataset
custom_dataset = CustomDataset(data, labels)
tensor_dataset = TensorDataset(data, labels)

# Create a DataLoader
dataloader = DataLoader(custom_dataset, batch_size=2, shuffle=True, collate_fn=default_collate)

for batch in dataloader:
    print(batch)
    

## custom image dataset
# step 1: Import necessary libraries
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# step 2 Organize your image data in a directory. For instance:
# . Directory Structure
# data/
#     ├── images/
#     │   ├── img1.jpg
#     │   ├── img2.jpg
#     │   └── img3.jpg
#     └── labels.csv  # A CSV file containing image filenames and their corresponding labels

# Step 3: Create a custom Dataset class
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with image filenames and labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied to the images.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])  # Assuming first column is filename
        image = Image.open(img_name)
        label = self.annotations.iloc[idx, 1]  # Assuming second column is label

        if self.transform:
            image = self.transform(image)

        return image, label

# Step 4: Define transformations (optional)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

# Step 5: Create the Dataset and DataLoader
# csv_file = 'data/labels.csv'  # Path to your CSV file
# root_dir = 'data/images'  # Path to the image directory

# custom_dataset = CustomImageDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
# dataloader = DataLoader(custom_dataset, batch_size=2, shuffle=True)

# Step 6: Example of iterating through the DataLoader
# for images, labels in dataloader:
#     print("Images batch shape:", images.shape)
#     print("Labels batch shape:", labels.shape)


## custom text dataset
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

# Example text data
texts = ["Hello world", "This is a test", "PyTorch is great", "Let's create a text dataset", "How are you?"]
labels = [0, 1, 1, 0, 1]  # Binary labels for classification

# Step 1: Tokenization
def tokenize(text):
    return text.lower().split()  # Simple tokenization by splitting words

# Step 2: Build Vocabulary
def build_vocab(texts):
    all_words = [word for text in texts for word in tokenize(text)]
    word_counts = Counter(all_words)
    vocab = {word: i for i, (word, _) in enumerate(word_counts.items())}
    return vocab

# Step 3: Custom Dataset Class
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize and convert to indices
        tokenized_text = tokenize(text)
        text_indices = [self.vocab[word] for word in tokenized_text if word in self.vocab]
        
        return torch.tensor(text_indices, dtype=torch.long), torch.tensor(label, dtype=torch.float)

# Step 4: Create Vocabulary
vocab = build_vocab(texts)

# Step 5: Create Dataset and DataLoader
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

train_dataset = TextDataset(train_texts, train_labels, vocab)
test_dataset = TextDataset(test_texts, test_labels, vocab)

def custom_collate(batch):
    # Zip	zip(list1, list2)	rdd1.zip(rdd2)
    # Unzip	zip(*zipped_list)	rdd.map(lambda x: x[0]), x[1], etc.
    # batch is a lit of tuple, each tuple contains a text and a label
    print("Batch:", batch)
    data, labels = zip(*batch) # put all texts in one list, all labels in another list
    print("Data:", data)
    print (labels)
    # Find max length for padding
    max_length = max(d.size(0) for d in data)
    # Pad sequences to the same length
    data = [torch.nn.functional.pad(d, (0, max_length - d.size(0))) for d in data]
    data = torch.stack(data)
    labels = torch.stack(labels)
    return data, labels

train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=custom_collate)

# Step 6: Example of iterating through the DataLoader
for batch in train_loader:
    text_indices, labels = batch
    print("Text Indices:", text_indices)
    print("Labels:", labels)

# Additional Example: Creating a TensorDataset
my_x = [np.array([[1.0, 2], [3, 4]]), np.array([[5., 6], [7, 8]])]  # a list of numpy arrays
my_y = [np.array([4.]), np.array([2.])]  # another list of numpy arrays (targets)

# Fixing the shape of my_x and my_y for tensor conversion
tensor_x = torch.Tensor([x.flatten() for x in my_x])  # flatten each array to create consistent shape
tensor_y = torch.Tensor(my_y)

# Create your dataset
my_dataset = TensorDataset(tensor_x, tensor_y)  
my_dataloader = DataLoader(my_dataset)  # Create your DataLoader

# Iterate through the DataLoader
for data, target in my_dataloader:
    print("Data:", data)
    print("Target:", target)

# Custom DataLoader with custom collate function
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
