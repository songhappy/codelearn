from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Sample dataset
texts = ["The quick brown fox", "jumps over the lazy dog", "A very long sentence indeed."]

# Tokenize and pad
def collate_fn(batch):
    return tokenizer(
        batch,
        padding=True,          # Pad to the longest sequence in the batch
        truncation=True,       # Truncate longer sequences
        max_length=128,        # Maximum length
        return_tensors="pt"    # Return PyTorch tensors
    )

# DataLoader
dataloader = DataLoader(texts, batch_size=2, collate_fn=collate_fn)

# Iterate through batches
for batch in dataloader:
    print(batch["input_ids"], batch["attention_mask"])
