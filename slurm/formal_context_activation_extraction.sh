#!/bin/bash
#SBATCH --job-name=ctx-formal-extract
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/ctx_formal_extract_%j.out

# Formal C-Direction Estimation extraction (one model per submission).
# Implements EXPERIMENT2_FORMAL_C_DIRECTION_PROTOCOL.md. Runs
# scripts/54_extract_formal_context_activations.py's REAL run (no
# --dry_run) for ONE of the 3 models, selected via MODEL_IDX. Phase 0's
# token-position gate (now 16 samples: 8 context + 8 canonical, including
# the 2 special-encoding mechanisms) runs first and aborts before any
# model weights load if it finds an anomaly.
#
# Submit once per model:
#   sbatch --export=MODEL_IDX=0 slurm/formal_context_activation_extraction.sh
#   sbatch --export=MODEL_IDX=1 slurm/formal_context_activation_extraction.sh
#   sbatch --export=MODEL_IDX=2 slurm/formal_context_activation_extraction.sh
#
# To dry-run one model first (CPU only, no model weights, no GPU needed):
#   srun --pty bash
#   . ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
#   cd ~/new_experiment
#   python3 scripts/54_extract_formal_context_activations.py \
#       --model_path /home/h24/baga0553/models/Llama-3.1-8B-Instruct \
#       --model_alias Meta-Llama-3.1-8B-Instruct --dry_run

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

python3 scripts/54_extract_formal_context_activations.py \
    --model_path  "$MODEL_PATH" \
    --model_alias "$MODEL_ALIAS" \
    --batch_size  1

echo "Done: $(date)"
