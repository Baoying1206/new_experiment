#!/bin/bash
#SBATCH --job-name=jb-tmpl-diffs
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/diffs_%j.out

# Stage-1 prerequisite: persist raw per-instruction, all-layer paired diffs
# (no generation, no WildGuard -- forward pass only, cheap). English-only
# first pass per the staged rollout plan.
#
# Submit with MODEL_IDX=0/1/2:
#   sbatch --export=MODEL_IDX=0 slurm/extract_paired_diffs.sh

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
LANGS=${LANGS:-en}

cd ~/new_experiment
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Model: $MODEL_ALIAS  Langs: $LANGS  Start: $(date)"

python scripts/18_extract_paired_diffs.py \
    --model_path  "$MODEL_PATH" \
    --model_alias "$MODEL_ALIAS" \
    --output_dir  output \
    --langs       "$LANGS" \
    --batch_size  8

echo "Done: $(date)"
