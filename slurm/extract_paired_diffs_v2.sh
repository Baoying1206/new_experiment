#!/bin/bash
#SBATCH --job-name=jb-tmpl-diffs-v2
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/diffs_v2_%j.out

# General-purpose submission for 18_extract_paired_diffs.py's confirmatory
# (572/200-instruction) scope -- forward-pass only, much cheaper than
# generation, no --time needed (already small relative to what succeeded
# without an explicit limit).
#
# Override via --export=MODEL_IDX=...,LANGS=...,SUFFIX=...,IDS_KEY=...
#
# Examples:
#   English direction set (300 ids, per model):
#     sbatch --export=MODEL_IDX=0,LANGS=en,SUFFIX=_full572,IDS_KEY=direction_ids slurm/extract_paired_diffs_v2.sh
#   English held-out replication (200 ids):
#     sbatch --export=MODEL_IDX=0,LANGS=en,SUFFIX=_full572,IDS_KEY=test_ids slurm/extract_paired_diffs_v2.sh
#   Cross-lingual direction set (100 ids, 5 languages in one job):
#     sbatch --export=MODEL_IDX=0,LANGS=zh,ar,th,yo,am,SUFFIX=_xling,IDS_KEY=cross_lingual_direction_ids slurm/extract_paired_diffs_v2.sh
#
# NOTE: sbatch --export splits on commas, so a LANGS value with multiple
# languages must use semicolons instead (converted to commas below) --
# same fix applied to ALPHAS in calibrate_injection_alpha.sh earlier:
#     sbatch --export=MODEL_IDX=0,LANGS="zh;ar;th;yo;am",SUFFIX=_xling,IDS_KEY=cross_lingual_direction_ids slurm/extract_paired_diffs_v2.sh

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
LANGS=${LANGS//;/,}
SUFFIX=${SUFFIX:-}
IDS_KEY_ARG=""
if [ -n "$IDS_KEY" ]; then
    IDS_KEY_ARG="--ids_key $IDS_KEY"
fi
MECHANISMS_ARG=""
if [ -n "$MECHANISMS" ]; then
    MECHANISMS=${MECHANISMS//;/,}
    MECHANISMS_ARG="--mechanisms $MECHANISMS"
fi

cd ~/new_experiment
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Model: $MODEL_ALIAS  Langs: $LANGS  Suffix: $SUFFIX  ids_key: ${IDS_KEY:-none}  mechanisms: ${MECHANISMS:-default(old 6)}  Start: $(date)"

python scripts/18_extract_paired_diffs.py \
    --model_path  "$MODEL_PATH" \
    --model_alias "$MODEL_ALIAS" \
    --output_dir  output \
    --langs       "$LANGS" \
    --suffix      "$SUFFIX" \
    $IDS_KEY_ARG \
    $MECHANISMS_ARG \
    --batch_size  8

echo "Done: $(date)"
