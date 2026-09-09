#!/bin/bash
#SBATCH --job-name=bt-bootstrap
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/behavioral_test_bootstrap_%j.out

# Behavioral Test bootstrap analysis (all 3 models, run AFTER all 3
# formal generation+judge runs in slurm/behavioral_test_formal.sh have
# completed). CPU-only, pure Python (no torch, no transformers) -- run via
# sbatch here only for convenience/logging; an interactive CPU node works
# identically. Refuses to run on a PILOT_NON_RESULT tree.
#
# Usage:
#   sbatch slurm/behavioral_test_bootstrap_analysis.sh

cd ~/new_experiment
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate

echo "Start: $(date)"

python3 scripts/57_behavioral_test_bootstrap_analysis.py \
    --formal_dir output/behavioral_test_formal \
    --model_aliases Qwen2.5-7B-Instruct,Meta-Llama-3.1-8B-Instruct,gemma-2-9b-it \
    --write_report output/behavioral_test_formal/behavioral_test_bootstrap_analysis.json

echo "Done: $(date)"
