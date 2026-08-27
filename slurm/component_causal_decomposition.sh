#!/bin/bash
#SBATCH --job-name=jb-tmpl-decomp
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/decomp_%j.out

# Decomposes template_direction into refusal-component / harmfulness-component
# (orthogonalized against refusal) / residual, and causally tests each via
# placebo-controlled injection -- see 21_component_causal_decomposition.py
# docstring for the full method (adapts Zhao et al. 2025's DiM+causal-steering
# +projection logic to test Wei et al.'s taxonomy at finer resolution than
# 09/19's raw geometric clustering test).
#
# Uses the ORIGINAL 75-instruction pilot's completions_{lang}.json (no
# --suffix support here yet) and alpha=2.0 as a starting point -- this has
# NOT been separately calibrated for component-level injection (component
# vectors are smaller than the full template_direction they're extracted
# from, since they're a projected fraction of it), so watch generation
# coherence in the log before trusting these numbers, same as any first run
# of a new injection setup.
#
# Submit with MODEL_IDX=0/1/2:
#   sbatch --export=MODEL_IDX=0 slurm/component_causal_decomposition.sh

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
REFUSAL_DIR_ROOT=${REFUSAL_DIR_ROOT:-/home/h24/baga0553/experiment_thesis/output/jailbreak_analysis}

MODEL_IDX=${MODEL_IDX:-0}
MODEL_PATH=${MODEL_PATHS[$MODEL_IDX]}
MODEL_ALIAS=${MODEL_ALIASES[$MODEL_IDX]}

cd ~/new_experiment
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Model: $MODEL_ALIAS  Start: $(date)"

python scripts/21_component_causal_decomposition.py \
    --model_path       "$MODEL_PATH" \
    --model_alias      "$MODEL_ALIAS" \
    --output_dir       output \
    --refusal_dir_root "$REFUSAL_DIR_ROOT" \
    --lang             en \
    --mechanisms       refusal_suppression,fictional_framing \
    --alpha            2.0 \
    --n_samples        40 \
    --batch_size       8

echo "Done: $(date)"
