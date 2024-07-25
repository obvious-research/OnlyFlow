#!/bin/bash

#SBATCH --job-name=onlyflow
#SBATCH --partition=gpu_p2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=24

#SBATCH --account=fkc@v100
#SBATCH --qos=qos_gpu-t3

#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --time=6:00:00
#SBATCH --output=outputs/output_%j/slurm/%x_%j_node_%n_task_%t.out
#SBATCH --error=outputs/output_%j/slurm/%x_%j_node_%n_task_%t.err

# Nettoyage des modules charges en interactif et herites par defaut
module purge

# Chargement des modules
module load anaconda-py3

conda activate onlyflow

export MASTER_PORT=$((49152 + RANDOM % 16384))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

set -x

srun training/dist_run.sh training/train.py +presets=zay/v100-8 $@ 8
