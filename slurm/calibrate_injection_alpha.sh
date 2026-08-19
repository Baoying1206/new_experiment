#!/bin/bash
#SBATCH --job-name=jb-tmpl-calib
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/calib_%j.out

# Phase 0 calibration, now with a placebo comparison at every alpha (not just
# the template direction) -- Qwen's Phase 1 run showed placebo inducing a
# surprisingly high bypass rate at alpha=2.0 (0.35-0.65 across languages),
# which is NOT what Arditi et al.'s magnitude-matched random-direction control
# convention predicts (a random/placebo direction at the same magnitude
# should NOT replicate the effect). So the alpha to pick is whichever gives
# the largest template-vs-placebo GAP with coherent generation, not just the
# alpha with the highest raw template induced_bypass_rate.
#
# Alpha ranges are lower for Gemma than Qwen/Llama, because Gemma already
# needed ~4x lower alpha than Qwen/Llama in the (different, unit-normalized)
# jailbreak_vector defense experiment (alpha=5 vs alpha=20) -- untested
# whether that ratio carries over to this raw-magnitude convention, but it's
# the only prior evidence available, so start conservative for Gemma.
#
# Submit with MODEL_IDX=0/1/2:
#   sbatch --export=MODEL_IDX=0 slurm/calibrate_injection_alpha.sh
#
# To override the alpha sweep, use ALPHAS with SEMICOLONS, not commas --
# sbatch's --export splits on commas itself, so a comma-separated value
# passed via --export gets silently truncated at the first comma:
#   sbatch --export=MODEL_IDX=2,ALPHAS="1.0;1.5;2.0;2.5" slurm/calibrate_injection_alpha.sh

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
ALPHA_RANGES=(
    "0.5,1.0,1.5,2.0"
    "0.5,1.0,1.5,2.0"
    "0.25,0.5,0.75,1.0"
)

MODEL_IDX=${MODEL_IDX:-0}
MODEL_PATH=${MODEL_PATHS[$MODEL_IDX]}
MODEL_ALIAS=${MODEL_ALIASES[$MODEL_IDX]}
ALPHAS=${ALPHAS:-${ALPHA_RANGES[$MODEL_IDX]}}
ALPHAS=${ALPHAS//;/,}  # allow semicolon-separated override (commas break --export)

cd ~/new_experiment
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Model: $MODEL_ALIAS  Alphas: $ALPHAS  Start: $(date)"

python scripts/10a_calibrate_injection_alpha.py \
    --model_path      "$MODEL_PATH" \
    --model_alias     "$MODEL_ALIAS" \
    --output_dir      output \
    --lang            en \
    --mechanism       refusal_suppression \
    --n_samples       10 \
    --alphas          "$ALPHAS" \
    --batch_size      8 \
    --max_new_tokens  200

echo "Done: $(date)"
