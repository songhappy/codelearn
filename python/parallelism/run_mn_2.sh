#!/bin/bash
#PBS -N hello-mn
#PBS -q debug-scaling                 
#PBS -l select=2:system=polaris:ncpus=8:ngpus=2   
#PBS -l walltime=00:10:00
#PBS -l filesystems=home:eagle 
#PBS -A Intel
#PBS -o helloworld_mnode.log
#PBS -e helloworld_mnode.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

# ===== Conda env =====
source /home/songhappy/miniconda3/etc/profile.d/conda.sh
conda activate peft

# ===== Tunables =====
export GPUS_PER_NODE="${GPUS_PER_NODE:-2}"   # GPUs per node (override on qsub if needed)
export MASTER_PORT="${MASTER_PORT:-29500}"
export TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL:-INFO}"

# ===== Derive distributed params from PBS =====
export NNODES=$(uniq "$PBS_NODEFILE" | wc -l)
export NUM_MACHINES="$NNODES"   # <— define it
export MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
 
echo "[DIST] NNODES=$NNODES  GPUS_PER_NODE=$GPUS_PER_NODE  WORLD_SIZE=$WORLD_SIZE"
echo "[DIST] MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "===== PBS Nodefile ====="; uniq "$PBS_NODEFILE"; echo "========================"


# using accelerate
# mpiexec -n "${NNODES}" -ppn 1 -hostfile "${PBS_NODEFILE}" bash -lc '
#   set -euo pipefail
#   source /home/songhappy/miniconda3/etc/profile.d/conda.sh
#   conda activate peft

#   # Use PMI_RANK if available, else OpenMPI’s rank, else 0
#   NODE_RANK=${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}

#   echo "[NODE] $(hostname -s) node_rank=$NODE_RANK  -> ${MASTER_ADDR}:${MASTER_PORT}"

#   accelerate launch \
#     --num_machines ${NNODES} \
#     --machine_rank ${NODE_RANK} \
#     --main_process_ip ${MASTER_ADDR} \
#     --main_process_port ${MASTER_PORT} \
#     --num_processes ${GPUS_PER_NODE} \
#     train_helloworld.py
# '

# useing torchrun
mpiexec -n 2 -ppn 1 -hostfile "${PBS_NODEFILE}" bash -lc '
  source /home/songhappy/miniconda3/etc/profile.d/conda.sh
  conda activate peft

  NODE_RANK=${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}

  torchrun \
    --nnodes ${NNODES} \
    --nproc_per_node ${GPUS_PER_NODE} \
    --node_rank \$NODE_RANK \
    --rdzv_backend c10d \
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    train_helloworld.py
'
