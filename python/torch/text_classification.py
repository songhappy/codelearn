import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

# Example text data
texts = ["Hello world", "This is a test", "PyTorch is great", "Let's create a text dataset", "How are you?", 
         "chatGPT is a transformer-based language model", "It uses the GPT-2 architecture", "The model is trained on a large corpus of text data", "The model can generate human-like text", "The model is fine-tuned for specific tasks"]
labels = [0, 1, 1, 0, 1, 1, 0, 1, 0, 1]  # Binary labels for classification

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
    data, labels = zip(*batch)
    # Find max length for padding
    max_length = max(d.size(0) for d in data)
    print("max_length", max_length)
    # Pad sequences to the same length
    data = [torch.nn.functional.pad(d, (0, max_length - d.size(0))) for d in data]
    data = torch.stack(data)
    labels = torch.stack(labels)
    print(data, labels)
    return data, labels

train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=custom_collate)

# Step 6: Define the Neural Network Model
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Binary classification
        
    def forward(self, x):
        # Average the embeddings for the words in the text
        x = self.embedding(x)
        x = x.mean(dim=1)  # Average pooling
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Step 7: Instantiate the model, define loss and optimizer
embedding_dim = 8
hidden_dim = 4
model = TextClassifier(vocab_size=len(vocab), embedding_dim=embedding_dim, hidden_dim=hidden_dim)

criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 8: Train the Model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        text_indices, labels = batch
        
        # Forward pass
        outputs = model(text_indices)  # Convert indices to float for BCELoss
        loss = criterion(outputs.squeeze(), labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 9: Evaluate the Model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch in DataLoader(test_dataset, batch_size=1, collate_fn=custom_collate):
        text_indices, labels = batch
        outputs = model(text_indices)
        predicted = (outputs.squeeze() > 0.5).float()  # Apply threshold
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f'Accuracy: {100 * correct / total:.2f}%')