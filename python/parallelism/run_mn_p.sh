#!/bin/bash
#PBS -N hello-mn
#PBS -q debug-scaling                 
#PBS -l select=2:system=polaris:ngpus=4
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
PYTHON=$(which python)

# ===== Tunables =====
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"   # GPUs per node (override on qsub if needed)
export MASTER_PORT="${MASTER_PORT:-29500}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL:-INFO}"

# ===== Derive distributed params from PBS =====
export NNODES=$(uniq "$PBS_NODEFILE" | wc -l)
export MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
 
echo "[DIST] NNODES=$NNODES  GPUS_PER_NODE=$GPUS_PER_NODE  WORLD_SIZE=$WORLD_SIZE"
echo "[DIST] MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "===== PBS Nodefile ====="; uniq "$PBS_NODEFILE"; echo "========================"

# # using mpiexec only a train_helloworld.py works well
# mpiexec --envall -n ${WORLD_SIZE} -ppn ${GPUS_PER_NODE} \
#     env MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" WORLD_SIZE="$WORLD_SIZE" \
#     $PYTHON train_helloworld.py

# using mpiexec only a train_fsdp2.py test works well
mpiexec --envall -n ${WORLD_SIZE} -ppn ${GPUS_PER_NODE} \
  bash -lc '
    # MPI → PyTorch env
    export RANK=${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}
    export LOCAL_RANK=$(( RANK % '"$GPUS_PER_NODE"' ))

    echo "[LAUNCH] host=$(hostname -s) RANK=$RANK LOCAL_RANK=$LOCAL_RANK WORLD_SIZE=$WORLD_SIZE"

    '"$PYTHON"' train_fsdp2.py
  '

# # using torchrun working well on both train_helloworld.py and train_fsdp2.py
# mpiexec -n "${NNODES}" -ppn 1 -hostfile "${PBS_NODEFILE}" bash -lc '
#   source /home/songhappy/miniconda3/etc/profile.d/conda.sh
#   conda activate peft
#   NODE_RANK=${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}
#   torchrun \
#     --nnodes ${NNODES} \
#     --nproc_per_node ${GPUS_PER_NODE} \
#     --node_rank  ${NODE_RANK} \
#     --rdzv_backend c10d \
#     --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
#     train_helloworld.py
# '

# # using accelerate working  well on both train_helloworld.py and train_fsdp2.py
# mpiexec -n "${NNODES}" -ppn 1 -hostfile "${PBS_NODEFILE}" bash -lc '
#   set -euo pipefail
#   source /home/songhappy/miniconda3/etc/profile.d/conda.sh
#   conda activate peft

#   # Use PMI_RANK if available, else OpenMPI’s rank, else 0
#   NODE_RANK=${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}
#   accelerate launch \
#     --num_machines ${NNODES} \
#     --machine_rank ${NODE_RANK} \
#     --main_process_ip ${MASTER_ADDR} \
#     --main_process_port ${MASTER_PORT} \
#     --num_processes ${WORLD_SIZE} \
#     train_helloworld.py
# '

# # --- Wrapper launched once per node via MPI working well
# cat > node_launch.sh <<'WRAP'
# #!/bin/bash
# set -euo pipefail
# MACHINE_RANK=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}
# accelerate launch \
#   --num_processes "${WORLD_SIZE}" \
#   --num_machines "$NUM_MACHINES" \
#   --machine_rank "${MACHINE_RANK}" \
#   --main_process_ip "${MASTER_ADDR}" \
#   --main_process_port "${MASTER_PORT}" \
#   --mixed_precision "no" \
#   train_helloworld.py
# WRAP
# chmod +x node_launch.sh

# # --- Launch one wrapper per node ---
# # The -ppn 1 ensures a single wrapper runs on each node; mpiexec comes from your MPI stack.
# mpiexec -n "${NUM_MACHINES}" -ppn 1 bash ./node_launch.sh