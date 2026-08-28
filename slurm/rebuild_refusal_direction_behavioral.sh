#!/bin/bash
#SBATCH --job-name=jb-refusal-v3
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/refusal_v3_%j.out

# Rebuilds refusal_direction with the paper's actual refused-vs-accepted
# contrast (26_rebuild_refusal_direction_behavioral.py), replacing the
# harmful-vs-harmless simplification used in
# 23_extract_reference_directions.py. Needs generation + WildGuard, so this
# is a real GPU cost (not just forward passes like the earlier v2 rebuild).
#
# ALWAYS run DRY_RUN=1 first and check the refused/accepted class balance
# printed in the log -- if accepted count is too small (<10), do not
# proceed to the real run without reconsidering n_axis.
#
# Override via --export=MODEL_IDX=...,DRY_RUN=1|0,CONFIRMED=1
#   Dry run:   sbatch --export=MODEL_IDX=0,DRY_RUN=1 slurm/rebuild_refusal_direction_behavioral.sh
#   Real run:  sbatch --export=MODEL_IDX=0,DRY_RUN=0,CONFIRMED=1 slurm/rebuild_refusal_direction_behavioral.sh

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
DRY_RUN=${DRY_RUN:-1}
CONFIRMED=${CONFIRMED:-0}

MODE_ARGS="--dry_run"
if [ "$DRY_RUN" = "0" ]; then
    MODE_ARGS=""
    if [ "$CONFIRMED" = "1" ]; then
        MODE_ARGS="--confirmed"
    fi
fi

cd ~/new_experiment
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Model: $MODEL_ALIAS  DRY_RUN=$DRY_RUN  CONFIRMED=$CONFIRMED  Start: $(date)"

python scripts/26_rebuild_refusal_direction_behavioral.py \
    --model_path  "$MODEL_PATH" \
    --model_alias "$MODEL_ALIAS" \
    --output_dir  output \
    --lang        en \
    $MODE_ARGS

echo "Done: $(date)"
