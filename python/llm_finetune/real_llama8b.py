import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from torch.distributed._composable.fsdp import fully_shard


#  Called on each process (each device will run one instance)
def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("xpu:xccl", rank=rank, world_size=world_size)  # Setup FSDP comms
    torch.xpu.set_device(rank)  #  Bind this process to one XPU
    return rank, torch.device(f"xpu:{rank}")


# ✨ Prompts are constructed in a consistent instruction format
def format_prompt(example):
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""


# 🔠 Tokenize and format the inputs for language modeling
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


def main():
    # 🔧 Setup distributed environment
    rank, device = setup_distributed()

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_path = "/home/guoqiong/models/Meta-Llama-3.1-8B-Instruct/"
    #model_path = "/home/guoqiong/models/Llama-3.3-70B-Instruct/"

    #  Tokenizer (shared vocab across all devices)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    #  Load base model (each process loads full model weights)
    print(f"[Rank {rank}] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": rank}  # Ensures loading on the correct XPU
    )

    #  Add LoRA adapters (lightweight trainable layers)
    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)

    #  Fully shard all trainable layers with FSDP2 (composable)
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear):
            fully_shard(module)

    #  Shard the outermost model (wrap stragglers)
    fully_shard(model)

    #  Load and tokenize dataset (each rank has full access)
    raw_dataset = load_dataset("tatsu-lab/alpaca", split="train[:500]")
    tokenized_dataset = raw_dataset.map(lambda x: tokenize_prompt(x, tokenizer))

    #  Ensure unique shard per GPU with DistributedSampler
    sampler = DistributedSampler(tokenized_dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True)
    dataloader = DataLoader(
        tokenized_dataset,
        sampler=sampler,  #  Ensures unique samples per rank
        batch_size=1,
        collate_fn=default_data_collator,
    )

    #  Optimizer (LoRA parameters only)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    # Training Loop
    model.train()
    for epoch in range(1):
        sampler.set_epoch(epoch)  #  Ensures proper shuffling across epochs
        for step, batch in enumerate(dataloader):
            inputs = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"[Rank {rank}] Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

    # 🧹 Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()