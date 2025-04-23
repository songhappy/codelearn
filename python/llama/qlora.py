import torch
from torchtune.modules.peft import LoRALinear

torch.set_default_device("xpu")
lora_linear = LoRALinear(512, 512, rank=8, alpha=0.1, quantize_base=False)
print(torch.xpu.memory_allocated()) # 1,081,344 bytes
del lora_linear

torch.xpu.empty_cache()
qlora_linear = LoRALinear(512, 512, rank=8, alpha=0.1, quantize_base=True)
print(torch.xpu.memory_allocated())  # 177,152 bytes
del qlora_linear

