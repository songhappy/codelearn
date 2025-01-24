from llama_simplified import SimplifiedLLaMAModel, TokenDataset, train
import torch
import torch._dynamo as dynamo

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

if __name__ == '__main__':

    vocab_size = 10000   # Number of tokens in the vocabulary
    embed_size = 1024     # Embedding size
    num_heads = 8            # Number of attention heads
    num_layers = 6       # Number of transformer layers
    ff_hidden_size = 2048 # Feedforward hidden size
    max_length = 100     # Maximum length of the input sequence
    dropout = 0.1        # Dropout rate
    max_generate_length = 20  # Maximum number of tokens to generate
    batch_size = 32      # Batch size for training
    num_epochs = 2      # Number of epochs to train
    learning_rate = 1e-4 # Learning rate
    
    if torch.xpu.is_available():
        device = "xpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    device = torch.device(device)

    # Initialize the simplified LLaMA-like model
    @dynamo.optimize("inductor") 
    def optimize_model():
        model = SimplifiedLLaMAModel(embed_size, num_layers, num_heads, ff_hidden_size, vocab_size, max_length, dropout)
        model.to(device)
        return model

    
    opt_model1 = optimize_model()

    model2 = SimplifiedLLaMAModel(embed_size, num_layers, num_heads, ff_hidden_size, vocab_size, max_length, dropout)
    opt_model2 = torch.compile(model2)
    opt_model2.to(device)

    model3 = SimplifiedLLaMAModel(embed_size, num_layers, num_heads, ff_hidden_size, vocab_size, max_length, dropout)
    model3.to(device)

    data = torch.randint(0, vocab_size, (1000, max_length))  # 1000 sequences of length max_length
    targets = torch.cat([data[:, 1:], torch.zeros(1000, 1, dtype=torch.long)], dim=1)  # Shifted target sequences

    # Create a dataset and dataloader
    dataset = TokenDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token (if any)
    optimizer1 = optim.Adam(opt_model1.parameters(), lr=learning_rate)
    optimizer2 = optim.Adam(opt_model2.parameters(), lr=learning_rate)
    optimizer3 = optim.Adam(model3.parameters(), lr=learning_rate)

    # Move model to device (GPU or CPU)

    # Training loop
    import time
    time_s1 = time.time()
    for epoch in range(num_epochs):
        epoch_loss = train(opt_model1, dataloader, optimizer1, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    time_e1 = time.time()

    time_s2 = time.time()
    for epoch in range(num_epochs):
        epoch_loss = train(opt_model2, dataloader, optimizer2, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    time_e2 = time.time()

    time_s3 = time.time()
    for epoch in range(num_epochs):
        epoch_loss = train(model3, dataloader, optimizer3, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    time_e3 = time.time()

    print(time_e1 - time_s1)
    print(time_e2 - time_s2)
    print(time_e3 - time_s3)


    @dynamo.optimize("inductor")
    def foo1(x, y):
        a = torch.sin(x)
        b = torch.cos(y)
        return a + b

    def foo2(x, y):
        a = torch.sin(x)
        b = torch.cos(y)
        return a + b
    opt_foo2 = torch.compile(foo2)
    
 
    a, b = torch.randn(10, 10), torch.randn(10, 10)

    time_s1 = time.time()
    for i in range(1000):
        x = foo1(a, b)
    time_e1 = time.time()

    time_s2 = time.time()
    for i in range(1000):
        x = opt_foo2(a, b)
    time_e2 = time.time()
    
    print(time_e1 - time_s1)
    print(time_e2 - time_s2)


