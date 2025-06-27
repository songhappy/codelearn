import os
import gc
import sys
import json
import torch
import subprocess
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from torch.distributed._composable.fsdp import fully_shard
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from myutils import get_gpu_memory_used_from_nvidia_smi

# Auto device selection
if torch.xpu.is_available():
    torch_device = torch.xpu
    backend = "xpu:xccl"
    device_type = "xpu"
elif torch.cuda.is_available():
    torch_device = torch.cuda
    backend = "nccl"
    device_type = "cuda"
else:
    torch_device = torch.device("cpu")
    backend = "gloo"
    device_type = "cpu"


# Memory print (XPU)
def get_xpu_memory_used_from_xpu_smi(tag, device_id=0):
    if device_type != "xpu":
        return
    try:
        result = subprocess.run(
            ["xpu-smi", "stats", "-d", str(device_id), "-j"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        stats = json.loads(result.stdout)
        tile_level = stats.get("tile_level", [])
        total_mem_mb = sum(metric["value"] for tile in tile_level for metric in tile.get("data_list", []) if metric["metrics_type"] == "XPUM_STATS_MEMORY_USED")
        return (f"[{tag}] xpu-smi memory used (device {device_id}): {total_mem_mb / 1024:.2f} GB")
    except Exception as e:
        return (f"[{tag}] xpu-smi error (device {device_id}): {e}")

# Memory print (CUDA)
def get_cuda_memory_used_from_nvidia_smi(tag, device_id=0):
    if device_type != "cuda":
        return
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader", "-i", str(device_id)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        used_mem_mb = float(result.stdout.strip())
        return(f"[{tag}] nvidia-smi memory used (device {device_id}): {used_mem_mb / 1024:.2f} GB")
    except Exception as e:
        return(f"[{tag}] nvidia-smi error (device {device_id}): {e}")


def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if device_type in ["xpu", "cuda"]:
        device = torch.device(f"{device_type}:{rank}")
        torch_device.set_device(device)
        return rank, device
    return rank, torch_device


def format_prompt(example):
    return f"""### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"""


def tokenize_prompt(example, tokenizer, max_length=512):
    prompt = format_prompt(example)
    tokenized = tokenizer(prompt, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    tokenized["labels"] = tokenized["input_ids"].clone()
    return {k: v.squeeze(0) for k, v in tokenized.items()}


def main():
    rank, device = setup_distributed()

    model_path = "/lus/grand/projects/Intel/models/Llama-3.1-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"[Rank {rank}] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": rank}
    )

    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.to(torch.bfloat16)

    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, LlamaDecoderLayer):
            fully_shard(module)
    fully_shard(model)

    raw_dataset = load_dataset("tatsu-lab/alpaca", split="train[:200]")
    tokenized_dataset = raw_dataset.map(lambda x: tokenize_prompt(x, tokenizer))

    sampler = DistributedSampler(tokenized_dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True)
    dataloader = DataLoader(tokenized_dataset, sampler=sampler, batch_size=1, collate_fn=default_data_collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    model.train()
    for epoch in range(1):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            inputs = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            memory_log = (
                get_xpu_memory_used_from_xpu_smi(f"Epoch {epoch}, Step {step}", device.index)
                if device_type == "xpu"
                else get_gpu_memory_used_from_nvidia_smi(f"Epoch {epoch}, Step {step}", device.index)
            )
            print(f"[Rank {rank}] Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}, {memory_log}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
