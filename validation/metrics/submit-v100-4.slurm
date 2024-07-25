#!/bin/bash

#SBATCH --job-name=flowctrl-metrics
#SBATCH --partition=gpu_p13
#SBATCH -C v100-32g
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10

#SBATCH --array=0-399%400

#SBATCH --account=fkc@v100
#SBATCH --qos=qos_gpu-t3

#SBATCH --hint=nomultithread
#SBATCH --time=4:00:00
#SBATCH --output=outputs/output_%A/slurm/%x_%A_%a_node_%n_task_%t.out
#SBATCH --error=outputs/output_%A/slurm/%x_%A_%a_node_%n_task_%t.err

# Nettoyage des modules charges en interactif et herites par defaut
module purge

# Chargement des modules
module load anaconda-py3

conda activate flowctrl

export MASTER_PORT=$((49152 + RANDOM % 16384))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)



set -x
export PYTHONPATH=$PYTHONPATH:$PWD
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

OUT_DIR="outputs/output_$SLURM_ARRAY_JOB_ID/artifacts"

srun python3 validation/metrics/validation_metrics.py +presets=zay/v100-4 hydra.run.dir=${OUT_DIR} dataset.video_size=512 validation.batch_size=2 dataset.shards=/lustre/fsn1/projects/rech/fkc/uhx75if/webvid/webvid_uw_desc_val.json resume="/lustre/fswork/projects/rech/fkc/uhx75if/flowctrl/outputs/output_1907439/artifacts/checkpoints/checkpoint-step-15500.ckpt"
