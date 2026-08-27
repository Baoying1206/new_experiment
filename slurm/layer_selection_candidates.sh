#!/bin/bash
#SBATCH --job-name=jb-layer-cand
#SBATCH --partition=cpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/layer_cand_%j.out

# CPU-only Decision 2 layer-selection candidates (22_layer_selection_candidates.py).
# Requires paired_diffs_{lang}{suffix}_validation_ids.pt to already exist
# (from 18_extract_paired_diffs.py --ids_key validation_ids).
#
# Override via --export=MODEL_ALIAS=...,LANG=...,SUFFIX=...
#   sbatch --export=MODEL_ALIAS=Qwen2.5-7B-Instruct,LANG=en,SUFFIX=_full572 slurm/layer_selection_candidates.sh

MODEL_ALIAS=${MODEL_ALIAS:-Qwen2.5-7B-Instruct}
LANG=${LANG:-en}
SUFFIX=${SUFFIX:-_full572}

cd ~/new_experiment
mkdir -p slurm/logs
VENV_PY=~/thesis_experiment/Multilingual-Refusal/venv/bin/python3
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Model: $MODEL_ALIAS  Lang: $LANG  Suffix: $SUFFIX  Start: $(date)"

"$VENV_PY" scripts/22_layer_selection_candidates.py \
    --model_alias "$MODEL_ALIAS" \
    --lang         "$LANG" \
    --suffix       "$SUFFIX"

echo "Done: $(date)"
