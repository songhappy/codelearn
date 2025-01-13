from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from ipex_llm.transformers import AutoModelForCausalLM

# Load pre-trained model and tokenizer
model_name = "/home/arda/intelWork/models/Llama-2-7b-chat-hf"  # Replace with the actual name of LLaMA 7B model from Hugging Face if different
tokenizer = LlamaTokenizer.from_pretrained(model_name)
# model = LlamaForCausalLM.from_pretrained(model_name, load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Example input text
input_text = "Once upon a time in a faraway land, there was a"

# Tokenize input text
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Generate output (text continuation)
# output_tokens = model.generate(inputs["input_ids"], max_length=50, num_return_sequences=1)

# # Decode and print the generated text
# generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
# print(generated_text)


# from transformers import LlamaForCausalLM, LlamaTokenizer
# import torch

# # Step 1: Load the pre-trained model and tokenizer
# tokenizer = LlamaTokenizer.from_pretrained(model_name)
# model = LlamaForCausalLM.from_pretrained(model_name)

# # Move the model to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# Step 2: Provide input text and tokenize
input_text = "The future of AI is"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Initialize the generated sequence with the input_ids
generated = inputs["input_ids"]

# Set the maximum length of the sequence to generate
max_length = 15

# Loop until the desired length is reached
for _ in range(max_length - inputs["input_ids"].shape[1]):
    # Step 3: Pass the current sequence through the model
    with torch.no_grad():
        outputs = model(generated)
    
    # Step 4: Extract logits and apply greedy search (select the token with the highest probability)
    print("outputs.logits.shape", outputs.logits.shape)
    next_token_logits = outputs.logits[:, -1, :]  # Get logits of the last token in the sequence
    next_token = torch.argmax(next_token_logits, dim=-1)  # Greedy: take the highest probability token
    
    print("next_token", next_token)
    print("next_token.shape", next_token.shape)
    # Append the predicted next token to the sequence
    generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    print("generated", generated)
    print("generated.shape", generated.shape)

    # Stop generation if the model outputs the end-of-sequence token (optional, based on LLaMA tokenizer)
    if next_token.item() == tokenizer.eos_token_id:
        break

# Step 5: Decode the generated sequence into text
generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)

# Tokenize the generated text into words
words = generated_text.split()

# Output the result
print(f"Generated Text: {generated_text}")
print(f"Tokenized Words: {words}")
