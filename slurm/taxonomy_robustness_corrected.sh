#!/bin/bash
#SBATCH --job-name=jb-taxo-corrected
#SBATCH --partition=cpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/taxo_corrected_%j.out

# CPU-only Exp1 re-test on the CORRECTED Wei taxonomy (32_taxonomy_robustness_corrected.py).
# Requires paired_diffs_{lang}_full572_corrected.pt (18_extract_paired_diffs.py
# --mechanisms matching _taxonomy_config.py's CORRECTED_REAL_MECHS).

MODEL_ALIASES=(
    "Qwen2.5-7B-Instruct"
    "Meta-Llama-3.1-8B-Instruct"
    "gemma-2-9b-it"
)
MODEL_IDX=${MODEL_IDX:-0}
MODEL_ALIAS=${MODEL_ALIASES[$MODEL_IDX]}

cd ~/new_experiment
mkdir -p slurm/logs
VENV_PY=~/thesis_experiment/Multilingual-Refusal/venv/bin/python3
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Model: $MODEL_ALIAS  Start: $(date)"

"$VENV_PY" scripts/32_taxonomy_robustness_corrected.py \
    --model_alias "$MODEL_ALIAS" \
    --lang         en \
    --suffix       _full572_corrected

echo "Done: $(date)"
