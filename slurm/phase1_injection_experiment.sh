#!/bin/bash
#SBATCH --job-name=jb-tmpl-phase1
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/phase1_%j.out

# Phase 1: causal sufficiency test with placebo control. 6 languages (2 per
# resource tier) x 3 mechanisms, alpha=2.0 (fixed by Phase 0 calibration on
# Qwen/en/refusal_suppression -- clean monotonic dose-response, no
# degeneration up to alpha=2.0).
#
# NOTE: alpha=2.0 was only calibrated on Qwen. Llama and Gemma have not been
# calibrated -- Gemma in particular needed a much lower alpha than Qwen/Llama
# for the (different) jailbreak_vector defense experiment
# (cross_lingual_defense.py: alpha=5 vs alpha=20), so alpha=2.0 may be too
# strong or too weak here. Watch the first few completions in the log for
# degenerate/repetitive text before trusting Gemma's numbers -- if generation
# looks broken, this needs its own Phase 0 calibration run first (adapt
# 10a_calibrate_injection_alpha.py with --model_path/--model_alias).
#
# Submit with MODEL_IDX=0/1/2:
#   sbatch --export=MODEL_IDX=0 slurm/phase1_injection_experiment.sh

MODEL_PATHS=(
    "/home/h24/baga0553/models/Qwen2.5-7B-Instruct"
    "/home/h24/baga0553/models/Llama-3.1-8B-Instruct"
    "/home/h24/baga0553/models/gemma-2-9b-it"
)
MODEL_ALIASES=(
    "Qwen2.5-7B-Instruct"
    "Meta-Llama-3.1-8B-Instruct"
    "gemma-2-9b-it"
)

MODEL_IDX=${MODEL_IDX:-0}
MODEL_PATH=${MODEL_PATHS[$MODEL_IDX]}
MODEL_ALIAS=${MODEL_ALIASES[$MODEL_IDX]}

cd ~/new_experiment
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Model: $MODEL_ALIAS  Start: $(date)"

python scripts/10b_phase1_injection_experiment.py \
    --model_path      "$MODEL_PATH" \
    --model_alias     "$MODEL_ALIAS" \
    --output_dir      output \
    --alpha           2.0 \
    --n_samples       40 \
    --batch_size      8 \
    --max_new_tokens  200

echo "Done: $(date)"
