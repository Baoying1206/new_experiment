#!/bin/bash
#SBATCH --job-name=jb-ref-dirs
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/ref_dirs_%j.out

# Rebuilds refusal_direction/harmfulness_direction at t_post/t_inst
# (Decision 1) from independent PolyRefuse train data (Decision 3). See
# scripts/23_extract_reference_directions.py's docstring for full details.
#
# ALWAYS run with DRY_RUN=1 first and review the printed positions before
# running for real (CONFIRMED=1) -- this is not optional.
#
# Override via --export=MODEL_IDX=...,DRY_RUN=1|0,CONFIRMED=1
#   Dry run (review positions, no GPU direction extraction saved):
#     sbatch --export=MODEL_IDX=0,DRY_RUN=1 slurm/extract_reference_directions.sh
#   Real extraction (only after reviewing the dry run):
#     sbatch --export=MODEL_IDX=0,DRY_RUN=0,CONFIRMED=1 slurm/extract_reference_directions.sh

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

python scripts/23_extract_reference_directions.py \
    --model_path  "$MODEL_PATH" \
    --model_alias "$MODEL_ALIAS" \
    --output_dir  output \
    --lang        en \
    $MODE_ARGS

echo "Done: $(date)"
