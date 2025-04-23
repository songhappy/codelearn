import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from modeling_deepseek_v3 import DeepseekV3ForCausalLM, DeepseekV3Config  # Ensure this is the correct import

# Create a significantly smaller configuration using DeepseekV3Config
small_config = DeepseekV3Config(
     vocab_size=100,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size = 256,
        num_hidden_layers=2,
        num_nextn_predict_layers=1,
        num_attention_heads=32,
        num_key_value_heads=32,
        n_shared_experts = 1,
        n_routed_experts = 8,
        ep_size = 1,
        routed_scaling_factor = 2.5,
        kv_lora_rank = 512,
        q_lora_rank = 1536,
        qk_rope_head_dim = 64,
        v_head_dim = 128,
        qk_nope_head_dim = 128,
        topk_method = 'noaux_tc',
        n_group = 8,
        topk_group = 4,
        num_experts_per_tok = 8,
        moe_layer_freq = 1,
        first_k_dense_replace = 3,
        norm_topk_prob = True,
        scoring_func = 'sigmoid',
        aux_loss_alpha = 0.001,
        seq_aux = True,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
)

# Instantiate the model with the small configuration
model = DeepseekV3ForCausalLM(small_config)
model = torch.compile(model)
model.to('xpu')  # Move model to XPU

# Create small dummy data
input_data = torch.randint(0, 20, (64, 128)).to('xpu')
labels = torch.randint(0, 20, (64, 128)).to('xpu')
print("Input Data:", input_data)
print("Labels:", labels)

# Define loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Forward pass
output = model(input_data, labels=labels)
loss = output.loss
print("Loss:", loss.item())

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()
torch.cuda.synchronize()

print("Training step completed.")
