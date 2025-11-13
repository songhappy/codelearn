#!/bin/bash
# ===== PBS directives =====
#PBS -N hell_sn
#PBS -q debug-scaling
#PBS -l walltime=00:10:00
#PBS -l select=1
#PBS -l filesystems=home:eagle
#PBS -A Intel
#PBS -o helloworld_singnode.log
#PBS -e helloworld_singnode.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

source /home/songhappy/miniconda3/etc/profile.d/conda.sh
conda activate peft

# Optional: pick a free port
MASTER_PORT="${MASTER_PORT:-29500}"
# Helpful logging
echo "Launching single-node training on 4 GPUs"
echo "MASTER_ADDR=127.0.0.1 MASTER_PORT=$MASTER_PORT"

# torchrun handles the process group setup; Accelerator will use NCCL + DDP
torchrun \
  --standalone \
  --nproc_per_node=4 \
  --master_port="$MASTER_PORT" \
  train_helloworld.py
