#!/bin/bash
#SBATCH --job-name=jb-tmpl-calib
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/calib_%j.out

# Phase 0: calibrate the injection alpha for the causal sufficiency test.
# Small, cheap run (Qwen only, en only, one mechanism, 10 prompts, 4 alphas)
# meant to be read by a human before committing to the full Phase 1 experiment.
#
# Submit (no MODEL_IDX needed -- Qwen only for calibration):
#   sbatch slurm/calibrate_injection_alpha.sh

cd ~/new_experiment
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Start: $(date)"

python scripts/10a_calibrate_injection_alpha.py \
    --model_path      /home/h24/baga0553/models/Qwen2.5-7B-Instruct \
    --model_alias     Qwen2.5-7B-Instruct \
    --output_dir      output \
    --lang            en \
    --mechanism       refusal_suppression \
    --n_samples       10 \
    --alphas          0.5,1.0,1.5,2.0 \
    --batch_size      8 \
    --max_new_tokens  200

echo "Done: $(date)"
