#!/bin/bash
#SBATCH --job-name=jb-tmpl-refgeo
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/refgeo_%j.out

# Geometric relationship between template_direction (this project) and
# refusal_direction / harmfulness_direction (experiment_thesis). Requires
# experiment_thesis/scripts/extract_jailbreak_vectors.py to already have been
# run for this model, with refusal_dir_{lang}.pt / harmfulness_dir_{lang}.pt
# present under $REFUSAL_DIR_ROOT/$MODEL_ALIAS/ for as many of
# en/zh/de/ko/ar/th/yo/sw/am as possible. Missing languages are skipped with
# a warning, not a hard failure -- check the log for which ones were skipped.
#
# Submit with MODEL_IDX=0/1/2:
#   sbatch --export=MODEL_IDX=0 slurm/refusal_geometry.sh

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
REFUSAL_DIR_ROOT=${REFUSAL_DIR_ROOT:-/home/h24/baga0553/experiment_thesis/output/jailbreak_analysis}

echo "Model: $MODEL_ALIAS  Start: $(date)"

cd ~/new_experiment
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

python scripts/09_refusal_geometry.py \
    --model_path       "$MODEL_PATH" \
    --model_alias      "$MODEL_ALIAS" \
    --output_dir       output \
    --refusal_dir_root "$REFUSAL_DIR_ROOT" \
    --batch_size        8

echo "Done: $(date)"
