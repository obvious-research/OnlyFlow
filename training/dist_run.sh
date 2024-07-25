#!/bin/bash

PT_SCRIPT=$1
GPUS_REQUESTED=${@: -1}
CONFIG="${@:2:$#-2}"


# if SLURM_GPUS_PER_NODE is set and GPUS_REQUESTED is set, then echo
if [ -n "${GPUS_REQUESTED}" ] && [ -n "${SLURM_GPUS_PER_NODE}" ]; then
    echo "GPUS_REQUESTED given but SLURM_GPUS_PER_NODE is set. Ignoring the number of GPUs given and using SLURM_GPUS_PER_NODE"
    NUM_GPUS=${SLURM_GPUS_PER_NODE}
else
    LOCAL_GPUS=$(nvidia-smi -L | wc -l)
    # if GPUS_REQUESTED is set
    if [ -n "${GPUS_REQUESTED}" ]; then
        if [ ${GPUS_REQUESTED} -gt ${LOCAL_GPUS} ]; then
            echo "$GPUS_REQUESTED GPUs requested is greater than $LOCAL_GPUS available GPUs on the node, exiting"
            exit 1
        fi
        NUM_GPUS=${GPUS_REQUESTED}
    # if SLURM_GPUS_PER_NODE is set
    elif [ -n "${SLURM_GPUS_PER_NODE}" ]; then
        NUM_GPUS=${SLURM_GPUS_PER_NODE}
    # if none of the above is set, then we default to the number of GPUs on the node based on nvidia-smi
    else
        NUM_GPUS=$LOCAL_GPUS
    fi
fi


NNODES=${SLURM_JOB_NUM_NODES:-1}

echo "Running on $NNODES nodes with $NUM_GPUS GPUs per node"

if [ ${NNODES} -gt 1 ]; then
    echo "Running distributed training"
    MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

else
    echo "Running single node training"
    MASTER_ADDR=$(hostname)
fi

MASTER_PORT=${MASTER_PORT:-$((49152 + RANDOM % 16384))}
# if SLURM_NODEID is not set, then we are not running
# in a SLURM environment so we set NODEID to 0
NODEID=${SLURM_NODEID:-0}
JOB_ID=${SLURM_JOB_ID:-0}

# add current folder to python path
export PYTHONPATH=$PYTHONPATH:$PWD

exec python3 -m torch.distributed.run \
--nnodes=$NNODES \
--node_rank=$NODEID \
--nproc_per_node=$NUM_GPUS \
--rdzv_backend=static \
--master_addr=${MASTER_ADDR} \
--master_port=${MASTER_PORT} \
${PT_SCRIPT} \
${CONFIG}



