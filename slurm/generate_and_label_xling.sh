#!/bin/bash
#SBATCH --job-name=jb-tmpl-gen-xling
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/generate_xling_%j.out
#SBATCH --time=12:00:00

# Cross-lingual validation: shared 200-instruction subset (data/splits.json's
# cross_lingual_ids) x 8 conditions = 1,600 rows/language, for the 8
# non-English pilot languages. Writes completions_{lang}_xling.json,
# distinct from the original 9-language pilot's completions_{lang}.json
# (75 instructions, kept as supplementary evidence, not overwritten).
#
# --time=12:00:00 covers all 8 languages in one job (~12,800 rows total,
# similar order of magnitude to the original 9-language x 75-instruction
# run which covered ~5,400 rows) -- check original logs and adjust if needed.
#
# Submit with MODEL_IDX=0/1/2:
#   sbatch --export=MODEL_IDX=0 slurm/generate_and_label_xling.sh

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
XLING_LANGS=(zh de ko ar th yo sw am)

MODEL_IDX=${MODEL_IDX:-0}
MODEL_PATH=${MODEL_PATHS[$MODEL_IDX]}
MODEL_ALIAS=${MODEL_ALIASES[$MODEL_IDX]}

echo "Model: $MODEL_ALIAS  Start: $(date)"

cd ~/new_experiment
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

for LANG in "${XLING_LANGS[@]}"; do
    echo "  [$LANG] Start: $(date)"
    python scripts/03_generate_and_label.py \
        --model_path     "$MODEL_PATH" \
        --model_alias    "$MODEL_ALIAS" \
        --lang           "$LANG" \
        --suffix          _xling \
        --output_dir     "output" \
        --batch_size     8 \
        --max_new_tokens 200
    echo "  [$LANG] Done: $(date)"
done

echo "All cross-lingual langs done: $(date)"
