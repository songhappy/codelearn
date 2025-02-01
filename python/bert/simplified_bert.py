import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

# Initialize Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Multi-Head Self-Attention Layer
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, "Hidden size must be divisible by number of heads"
        
        self.query_fc = nn.Linear(hidden_size, hidden_size)
        self.key_fc = nn.Linear(hidden_size, hidden_size)
        self.value_fc = nn.Linear(hidden_size, hidden_size)
        self.out_fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # Split the input into multiple heads
        Q = self.query_fc(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_fc(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_fc(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / self.head_dim ** 0.5
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        # Apply attention to the values
        attention_output = torch.matmul(attention_weights, V).transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        
        # Output projection
        return self.out_fc(attention_output)

# Feedforward Network
class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super(TransformerEncoderLayer, self).__init__()
        
        self.attn = MultiHeadAttention(hidden_size, num_heads)
        self.feedforward = FeedForward(hidden_size, intermediate_size)
        
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Multi-head attention with residual connection
        attn_output = self.attn(x, x, x)
        x = self.layer_norm1(x + self.dropout(attn_output))
        
        # Feedforward network with residual connection
        ff_output = self.feedforward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        
        return x

# Simplified BERT Model with Transformer Encoder
class SimpleBERT(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=128, num_heads=4, num_layers=4, intermediate_size=512):
        super(SimpleBERT, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(512, hidden_size)
        
        # Stack of Transformer Encoder layers
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)])
        
        self.mlm_head = nn.Linear(hidden_size, vocab_size)  # Prediction layer for MLM

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)

        # Add token and position embeddings
        token_embeddings = self.embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        
        embeddings = token_embeddings + position_embeddings
        
        # Pass through Transformer Encoder stack
        for layer in self.encoder_layers:
            embeddings = layer(embeddings)
        
        # Output logits (MLM prediction)
        logits = self.mlm_head(embeddings)
        return logits


# Dataset for MLM
class MaskedTextDataset(Dataset):
    def __init__(self, sentences, tokenizer, mask_prob=0.15, max_len=10):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        
        # Tokenize sentence
        tokens = self.tokenizer(sentence, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')['input_ids'][0]
        labels = tokens.clone()

        # Mask some tokens randomly
        for i in range(1, len(tokens) - 1):  # Avoid CLS and SEP tokens
            if random.random() < self.mask_prob:
                tokens[i] = self.tokenizer.mask_token_id

        return tokens, labels


# Sample dataset
sentences = [
    "I love programming with Python.",
    "Deep learning is amazing.",
    "Natural language processing is fun.",
    "Transformers are great for NLP tasks."
]

# Create Dataset and DataLoader
dataset = MaskedTextDataset(sentences, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Instantiate the model
model = SimpleBERT()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # Ignore pad tokens in loss calculation
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    
    for input_batch, target_batch in dataloader:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_batch)
        
        # Reshape outputs and targets for loss calculation
        logits = logits.view(-1, logits.size(-1))
        target_batch = target_batch.view(-1)
        
        # Calculate loss
        loss = criterion(logits, target_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training Complete!")


# Prediction function
def predict_masked_sentence(text):
    model.eval()
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=10, return_tensors='pt')['input_ids'].to(device)

    with torch.no_grad():
        logits = model(tokens)

    predictions = torch.argmax(logits, dim=-1)[0]
    predicted_tokens = tokenizer.convert_ids_to_tokens(predictions.cpu().numpy())
    
    return tokenizer.decode(predictions.cpu().numpy(), skip_special_tokens=True)


# Test on masked sentence
masked_sentence = "Transformers are [MASK] for NLP tasks."
print("Original Sentence:", masked_sentence)
print("Predicted Sentence:", predict_masked_sentence(masked_sentence))
