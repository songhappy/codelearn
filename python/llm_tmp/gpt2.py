import sys
from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
print(model)
# sys.exit()
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model.generate(**encoded_input)
print(output)
print(type(output))
result = tokenizer.decode(output[0])
print(result)

