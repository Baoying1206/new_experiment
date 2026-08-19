#!/bin/bash
#SBATCH --job-name=jb-tmpl-phase1
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/phase1_%j.out

# Phase 1: causal sufficiency test with placebo control. 6 languages (2 per
# resource tier) x 3 mechanisms, alpha=2.0 (fixed by Phase 0 calibration on
# Qwen/en/refusal_suppression -- clean monotonic dose-response, no
# degeneration up to alpha=2.0). Qwen only for this first pass.
#
# Submit:
#   sbatch slurm/phase1_injection_experiment.sh

cd ~/new_experiment
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Start: $(date)"

python scripts/10b_phase1_injection_experiment.py \
    --model_path      /home/h24/baga0553/models/Qwen2.5-7B-Instruct \
    --model_alias     Qwen2.5-7B-Instruct \
    --output_dir      output \
    --alpha           2.0 \
    --n_samples       40 \
    --batch_size      8 \
    --max_new_tokens  200

echo "Done: $(date)"
