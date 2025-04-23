python finetune_lora_il.py \
    --micro_batch_size 8 \
    --batch_size 128 \
    --base_model "/mnt/models/Llama-3.2-1B-Instruct/" \
    --data_path "yahma/alpaca-cleaned" \
    --output_dir "./ipex-llm-lora-alpaca" \
    --gradient_checkpointing True \
    --lora_target_modules "['k_proj', 'q_proj', 'o_proj', 'v_proj']"
