#!/bin/bash
#SBATCH --job-name=ctx-act-pilot
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/ctx_activation_pilot_%j.out

# Context Activation Pilot (Meta-Llama-3.1-8B-Instruct, direction_ids[:30],
# 16 context conditions + 4 canonical conditions = 20, 600 forward passes).
# Runs scripts/52_extract_context_activations_pilot.py's REAL pilot (no
# --dry_run). Phase 0's token-position gate (12 samples) runs first and
# aborts before any model weights load if it finds an anomaly -- if this
# job exits fast with a GATE VIOLATION, no GPU time was wasted on the
# forward passes.
#
# Usage:
#   sbatch slurm/context_activation_pilot.sh
#
# To run the CPU-only --dry_run instead (no model weights, no GPU needed),
# don't submit this via sbatch -- use an interactive CPU node as before:
#   srun --pty bash
#   . ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
#   python3 scripts/52_extract_context_activations_pilot.py \
#       --model_path /home/h24/baga0553/models/Llama-3.1-8B-Instruct --dry_run

cd ~/new_experiment
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Start: $(date)"

python3 scripts/52_extract_context_activations_pilot.py \
    --model_path /home/h24/baga0553/models/Llama-3.1-8B-Instruct \
    --batch_size 1

echo "Done: $(date)"
