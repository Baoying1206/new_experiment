#!/bin/bash
#SBATCH --job-name=jb-token-audit
#SBATCH --partition=cpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/token_audit_%j.out

# Token-position audit (scripts/audits/audit_token_positions.py) -- only
# loads tokenizers (AutoTokenizer.from_pretrained), never the model weights,
# so this is CPU-only and light. Required before any full GPU direction-
# extraction job using scripts/utils/token_positions.py (t_inst/t_post) --
# per the user's explicit instruction, do not skip the human-review step
# after this runs.

MODEL_PATHS=${MODEL_PATHS:-"/home/h24/baga0553/models/Qwen2.5-7B-Instruct,/home/h24/baga0553/models/Llama-3.1-8B-Instruct,/home/h24/baga0553/models/gemma-2-9b-it"}
MODEL_ALIASES=${MODEL_ALIASES:-"Qwen2.5-7B-Instruct,Meta-Llama-3.1-8B-Instruct,gemma-2-9b-it"}

cd ~/new_experiment
mkdir -p slurm/logs
VENV_PY=~/thesis_experiment/Multilingual-Refusal/venv/bin/python3
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Model paths: $MODEL_PATHS  Start: $(date)"

"$VENV_PY" scripts/audits/audit_token_positions.py \
    --model_paths   "$MODEL_PATHS" \
    --model_aliases "$MODEL_ALIASES"

echo "Done: $(date)"
