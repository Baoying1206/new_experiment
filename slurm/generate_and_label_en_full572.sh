#!/bin/bash
#SBATCH --job-name=jb-tmpl-gen-en572
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/generate_en572_%j.out

# English core experiment at full scale: 572 instructions x 8 conditions =
# 4,576 rows -- ~7.6x the original 75-instruction pilot's English run.
# Writes completions_en_full572.json, distinct from the original
# completions_en.json (kept as supplementary evidence, not overwritten).
#
# No explicit --time -- uses the partition/association default, same as the
# original generate_and_label.sh (9 languages x 75 instructions = 5,400 rows
# total), which completed successfully under that default. This English-only
# run is 4,576 rows, smaller than that already-proven job, so omitting
# --time should be safe. (An earlier version of this script set
# --time=12:00:00 explicitly, which exceeded this account's
# AssocMaxWallDurationPerJobLimit and left jobs stuck pending -- removed.)
#
# Submit with MODEL_IDX=0/1/2:
#   sbatch --export=MODEL_IDX=0 slurm/generate_and_label_en_full572.sh

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
    --model_path     "$MODEL_PATH" \
    --model_alias    "$MODEL_ALIAS" \
    --lang           en \
    --suffix          _full572 \
    --output_dir     "output" \
    --batch_size     8 \
    --max_new_tokens 200

echo "Done: $(date)"
