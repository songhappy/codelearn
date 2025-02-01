import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
import random

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define BERT Model
class BERT(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=768, num_heads=12, num_layers=6, intermediate_size=3072):  # Reduced layers for fast training
        super(BERT, self).__init__()
        
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(512, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=intermediate_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.mlm_head = nn.Linear(hidden_size, vocab_size)  # MLM prediction head

    def forward(self, input_ids):
        seq_length = input_ids.shape[1]
        positions = torch.arange(0, seq_length, dtype=torch.long).unsqueeze(0).to(input_ids.device)
        
        # Word + Position Embeddings
        embeddings = self.embeddings(input_ids) + self.position_embeddings(positions)
        
        # Pass through Transformer Encoder
        encoded = self.encoder(embeddings)
        
        # MLM Prediction
        logits = self.mlm_head(encoded)
        return logits


# Sample training data
sentences = [
    "I love machine learning.",
    "The cat sat on the mat.",
    "PyTorch makes deep learning easy.",
    "Transformers are powerful models.",
]

# Tokenization and Masking
def tokenize_and_mask(sentences, mask_prob=0.15):
    tokenized = []
    labels = []
    
    for sentence in sentences:
        tokens = tokenizer(sentence, padding="max_length", max_length=10, truncation=True, return_tensors="pt")["input_ids"][0]
        label = tokens.clone()
        
        # Apply random masking (except CLS and SEP)
        for i in range(1, len(tokens) - 1):  
            if random.random() < mask_prob:
                tokens[i] = tokenizer.mask_token_id  # Replace token with [MASK]
        
        tokenized.append(tokens)
        labels.append(label)

    return torch.stack(tokenized), torch.stack(labels)

# Convert sentences to tokenized format
inputs, targets = tokenize_and_mask(sentences)

# Initialize BERT
model = BERT()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # Ignore padding tokens
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Training Loop
num_epochs = 5
batch_size = 2

for epoch in range(num_epochs):
    model.train()
    
    for i in range(0, len(inputs), batch_size):
        input_batch = inputs[i : i + batch_size].to(device)
        target_batch = targets[i : i + batch_size].to(device)

        optimizer.zero_grad()
        
        # Forward pass
        output = model(input_batch)

        # Compute loss only for masked tokens
        mask = (input_batch == tokenizer.mask_token_id)
        output = output.view(-1, tokenizer.vocab_size)  # Reshape for loss
        target_batch = target_batch.view(-1)  # Flatten targets
        
        loss = criterion(output[mask.view(-1)], target_batch[mask.view(-1)])  # Only consider masked tokens
        
        # Backward pass
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete!")

# Function to predict masked text
def predict_masked_text(text):
    model.eval()
    
    # Tokenize input sentence
    tokens = tokenizer(text, padding="max_length", max_length=10, truncation=True, return_tensors="pt")["input_ids"].to(device)

    with torch.no_grad():
        logits = model(tokens)

    predictions = torch.argmax(logits, dim=-1)[0]

    # Replace only MASK tokens
    tokens = tokens[0].tolist()
    for i, token in enumerate(tokens):
        if token == tokenizer.mask_token_id:
            tokens[i] = predictions[i].item()

    # Decode only the relevant tokens, ignoring PAD tokens
    return tokenizer.decode(tokens, skip_special_tokens=True)

# Test on new sentence
masked_sentence = "Transformers are [MASK] models."
print("Original:", masked_sentence)
print("Predicted:", predict_masked_text(masked_sentence))
