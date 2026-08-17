#!/bin/bash
#SBATCH --job-name=jb-tmpl-gen
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/generate_and_label_%j.out

# Runs 03_generate_and_label.py for all three pilot languages (en, ko, yo),
# for one model. Mirrors experiment_thesis/slurm/baseline_inference.sh.
#
# Submit with MODEL_IDX=0/1/2:
#   sbatch --export=MODEL_IDX=0 slurm/generate_and_label.sh
#   sbatch --export=MODEL_IDX=1 slurm/generate_and_label.sh
#   sbatch --export=MODEL_IDX=2 slurm/generate_and_label.sh
#
# Prerequisite: templates/templates_yo.json must be real translations, not the
# [NEEDS REAL TRANSLATION] placeholders -- 03_generate_and_label.py will raise
# and stop rather than silently run on untranslated text.

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
PILOT_LANGS=(en zh de ko ar th yo sw am)

MODEL_IDX=${MODEL_IDX:-0}
MODEL_PATH=${MODEL_PATHS[$MODEL_IDX]}
MODEL_ALIAS=${MODEL_ALIASES[$MODEL_IDX]}

echo "Model: $MODEL_ALIAS  Start: $(date)"

cd ~/new_experiment
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

for LANG in "${PILOT_LANGS[@]}"; do
    echo "  [$LANG] Start: $(date)"
    python scripts/03_generate_and_label.py \
        --model_path     "$MODEL_PATH" \
        --model_alias    "$MODEL_ALIAS" \
        --lang           "$LANG" \
        --output_dir     "output" \
        --batch_size     8 \
        --max_new_tokens 200
    echo "  [$LANG] Done: $(date)"
done

echo "All pilot langs done: $(date)"
