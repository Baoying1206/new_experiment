#!/bin/bash
#SBATCH --job-name=jb-delta-rh
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/delta_rh_%j.out

# Experiment 2 core data: delta_R/delta_H extraction (25_extract_delta_r_h.py).
# Requires output_v2_dual_position/{model_alias}/refusal_dir_v2_{lang}.pt +
# harmfulness_dir_v2_{lang}.pt to already exist (23_extract_reference_directions.py
# --confirmed) and completions_{lang}{suffix}.json to exist for this model.
#
# ~2400 batch_size=1 forward passes (300 instructions x 8 conditions) --
# based on the reference-direction extraction's timing (256 passes in
# ~40s), expect roughly 6-7 minutes, no special --time needed.
#
# Override via --export=MODEL_IDX=...
#   sbatch --export=MODEL_IDX=0 slurm/extract_delta_r_h.sh

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

cd ~/new_experiment
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Model: $MODEL_ALIAS  Start: $(date)"

python scripts/25_extract_delta_r_h.py \
    --model_path  "$MODEL_PATH" \
    --model_alias "$MODEL_ALIAS" \
    --output_dir  output \
    --lang        en \
    --suffix      _full572 \
    --ids_key     direction_ids

echo "Done: $(date)"
