#!/bin/bash
#SBATCH --job-name=jb-tmpl-magbeh
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/magbeh_%j.out

# Direction-magnitude vs behavioral ΔASR correlation, for all 3 models.
# Requires pilot_results.json (for bypass_rates) to already exist for this
# model -- run analyze_pilot.sh first if not.
#
# Submit with MODEL_IDX=0/1/2:
#   sbatch --export=MODEL_IDX=0 slurm/magnitude_vs_behavior.sh

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

python scripts/08_magnitude_vs_behavior.py \
    --model_path  "$MODEL_PATH" \
    --model_alias "$MODEL_ALIAS" \
    --output_dir  output \
    --batch_size  8

echo "Done: $(date)"
