import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, default_data_collator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from torch.distributed._composable.fsdp import fully_shard
from accelerate import init_empty_weights
from safetensors import safe_open
from typing import Dict, Any
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed._tensor import distribute_tensor, DTensor


def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("xpu:xccl", rank=rank, world_size=world_size)
    torch.xpu.set_device(rank)
    return rank, torch.device(f"xpu:{rank}")


def format_prompt(example):
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""


def tokenize_prompt(example, tokenizer, max_length=512):
    prompt = format_prompt(example)
    tokenized = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return {k: v.squeeze(0) for k, v in tokenized.items()}


def load_full_safetensors_state_dict(model_path, prefix=""):
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_file, "r") as f:
        index = json.load(f)

    full_sd = {}
    for shard_file in set(index["weight_map"].values()):
        shard_path = os.path.join(model_path, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                full_sd[prefix + key] = f.get_tensor(key)
    return full_sd


def load_from_full_model_state_dict(
    model: nn.Module,
    full_sd: Dict[str, Any],
    device: torch.device,
    strict: bool = False,
    cpu_offload: bool = False,
    use_distributed_state_dict: bool = False,
):
    meta_sharded_sd = model.state_dict()

    sharded_sd = {}
    for param_name, full_tensor in full_sd.items():
        sharded_meta_param = meta_sharded_sd.get(param_name)
        full_tensor = full_tensor.to(sharded_meta_param.dtype).to(device)
        if not hasattr(sharded_meta_param, "device_mesh"):
            sharded_tensor = full_tensor
        else:
            sharded_tensor = distribute_tensor(
                full_tensor,
                sharded_meta_param.device_mesh,
                sharded_meta_param.placements,
            )
        if cpu_offload:
            sharded_tensor = sharded_tensor.cpu()
        sharded_sd[param_name] = nn.Parameter(sharded_tensor)
    return model.load_state_dict(sharded_sd, strict=strict, assign=True)


def main():
    rank, device = setup_distributed()
    model_path = "/home/guoqiong/models/Llama-3.3-70B-Instruct/"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    config = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    model = get_peft_model(model, lora_cfg)

    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear):
            fully_shard(module)
    fully_shard(model)

    prefix = "base_model.model."
    full_sd = load_full_safetensors_state_dict(model_path, prefix=prefix)
    load_from_full_model_state_dict(model, full_sd, device=device)

    meta_params = [n for n, p in model.named_parameters() if p.is_meta and "lora_" not in n]
    if meta_params:
        print(f"[Rank {rank}] ❌ Meta params remain (excluding LoRA): {meta_params}")
        raise RuntimeError("Some non-LoRA model parameters were not properly loaded.")
    else:
        print(f"[Rank {rank}] ✅ All base model parameters loaded successfully.")

    raw_dataset = load_dataset("tatsu-lab/alpaca", split="train[:100]")
    tokenized_dataset = raw_dataset.map(lambda x: tokenize_prompt(x, tokenizer))

    sampler = DistributedSampler(tokenized_dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True)
    dataloader = DataLoader(
        tokenized_dataset,
        sampler=sampler,
        batch_size=1,
        collate_fn=default_data_collator,
    )

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

            if step % 10 == 0:
                print(f"[Rank {rank}] Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
