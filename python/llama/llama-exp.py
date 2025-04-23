from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import torch.ao.quantization as quantization

# Local path where you downloaded the model
model_path = "/mnt/models/Llama-2-7b-hf"

# Load tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_path)

# Load model, forcing it to accept the bin format
model = LlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Adjust based on available resources
    device_map="auto",
    low_cpu_mem_usage=True,  # Helps with large models
    use_safetensors=False
)

model = model.to("xpu")
# Example prompt
prompt = "Once upon a time in a futuristic city,"
inputs = tokenizer(prompt, return_tensors="pt").to("xpu")

# Generate text
with torch.no_grad():
    output = model.generate(**inputs, max_length=100)

# Decode output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)


# Apply dynamic quantization (reduces memory and speeds up inference on CPU)
quantized_model = quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Target Linear layers for quantization
    dtype=torch.qint8  # Convert to int8 for efficiency
)

# Example prompt
prompt = "Once upon a time in a futuristic city,"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text using the quantized model
with torch.no_grad():
    output = quantized_model.generate(
        **inputs,
        max_length=100,  # Control output length
        temperature=0.7,  # Adjust creativity
        top_p=0.9,  # Nucleus sampling
        repetition_penalty=1.1  # Reduce repetition
    )

# Decode and print the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

