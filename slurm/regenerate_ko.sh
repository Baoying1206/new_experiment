#!/bin/bash
#SBATCH --job-name=jb-tmpl-ko-fix
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/regenerate_ko_%j.out

# One-off: regenerate ko completions with --overwrite after the
# instruction_hierarchy template fix (ko was stuck on the old "SYSTEM
# OVERRIDE" wording while every other language had already been updated).
#
# Submit with MODEL_IDX=0/1/2:
#   sbatch --export=MODEL_IDX=0 slurm/regenerate_ko.sh
#   sbatch --export=MODEL_IDX=1 slurm/regenerate_ko.sh
#   sbatch --export=MODEL_IDX=2 slurm/regenerate_ko.sh

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

echo "Model: $MODEL_ALIAS  Start: $(date)"

cd ~/new_experiment
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

python scripts/03_generate_and_label.py \
    --model_path  "$MODEL_PATH" \
    --model_alias "$MODEL_ALIAS" \
    --lang        ko \
    --output_dir  output \
    --batch_size  8 \
    --max_new_tokens 200 \
    --overwrite

echo "Done: $(date)"
