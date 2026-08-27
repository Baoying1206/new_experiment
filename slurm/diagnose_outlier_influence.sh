#!/bin/bash
#SBATCH --job-name=jb-tmpl-outlier
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/outlier_%j.out

# Checks whether template_direction's raw mean is dominated by a few
# high-magnitude per-instruction diff vectors. English-only first pass.
#
# Submit with MODEL_IDX=0/1/2, optionally LAYER=<int> (default n_layers//2):
#   sbatch --export=MODEL_IDX=0 slurm/diagnose_outlier_influence.sh
#   sbatch --export=MODEL_IDX=0,LAYER=14 slurm/diagnose_outlier_influence.sh

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
LAYER_ARG=""
if [ -n "$LAYER" ]; then
    LAYER_ARG="--layer $LAYER"
fi

cd ~/new_experiment
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Model: $MODEL_ALIAS  Layer override: ${LAYER:-none}  Start: $(date)"

python scripts/17_diagnose_outlier_influence.py \
    --model_path  "$MODEL_PATH" \
    --model_alias "$MODEL_ALIAS" \
    --output_dir  output \
    --langs       en \
    --batch_size  8 \
    $LAYER_ARG

echo "Done: $(date)"
