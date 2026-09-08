#!/bin/bash
#SBATCH --job-name=ctx-formal-analyze
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/ctx_formal_analyze_%j.out

# Formal C-Direction Estimation analysis (all 3 models, run AFTER all 3
# extractions in slurm/formal_context_activation_extraction.sh have
# completed). CPU-only (pure tensor math on already-extracted .pt files,
# no tokenizer, no model weights) -- submitted via sbatch here only for
# convenience/logging; an interactive srun CPU node works identically,
# same as the pilot's scripts/53 analysis.
#
# Usage:
#   sbatch slurm/formal_context_activation_analysis.sh

cd ~/new_experiment
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Start: $(date)"

python3 scripts/55_analyze_formal_context_activations.py \
    --formal_dir output/context_activations_formal \
    --model_aliases Qwen2.5-7B-Instruct,Meta-Llama-3.1-8B-Instruct,gemma-2-9b-it \
    --write_report output/context_activations_formal/formal_c_direction_analysis.json

echo "Done: $(date)"
