#!/bin/bash
#SBATCH --job-name=jb-ref-diag
#SBATCH --partition=cpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/ref_diag_%j.out

# CPU-only diagnostics on the v2 dual-position reference directions
# (24_reference_direction_diagnostics.py) -- cos(refusal, harmfulness) per
# layer plus norm sanity checks. Requires output_v2_dual_position/{model}/
# refusal_dir_v2_{lang}.pt and harmfulness_dir_v2_{lang}.pt to already exist
# (from 23_extract_reference_directions.py --confirmed).
#
# Override via --export=MODEL_ALIAS=...,LANG=...
#   sbatch --export=MODEL_ALIAS=Qwen2.5-7B-Instruct,LANG=en slurm/reference_direction_diagnostics.sh

MODEL_ALIAS=${MODEL_ALIAS:-Qwen2.5-7B-Instruct}
LANG=${LANG:-en}

cd ~/new_experiment
mkdir -p slurm/logs
VENV_PY=~/thesis_experiment/Multilingual-Refusal/venv/bin/python3
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Model: $MODEL_ALIAS  Lang: $LANG  Start: $(date)"

"$VENV_PY" scripts/24_reference_direction_diagnostics.py \
    --model_alias "$MODEL_ALIAS" \
    --lang         "$LANG"

echo "Done: $(date)"
