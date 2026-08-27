#!/bin/bash
#SBATCH --job-name=jb-taxo-robust
#SBATCH --partition=cpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/taxo_robust_%j.out

# CPU-only Exp1 analysis (19_taxonomy_robustness.py) -- no GPU/model loading
# needed, so this runs on the `cpu` partition (confirmed available via
# sinfo) rather than occupying a gpu node. Uses the venv's python via its
# full path rather than `source activate` -- on this cluster's login node,
# `source venv/bin/activate` sets VIRTUAL_ENV/PS1 but something resets PATH
# afterward so `python3`/`pip` still resolve to the system install; calling
# the venv's python3 binary directly sidesteps that (compute nodes may not
# have the same issue, but this is the safer, verified-working form).
#
# Override via --export=MODEL_ALIAS=...,LANG=...,SUFFIX=...
#   sbatch --export=MODEL_ALIAS=Qwen2.5-7B-Instruct,LANG=en,SUFFIX=_full572 slurm/taxonomy_robustness.sh

MODEL_ALIAS=${MODEL_ALIAS:-Qwen2.5-7B-Instruct}
LANG=${LANG:-en}
SUFFIX=${SUFFIX:-_full572}

cd ~/new_experiment
mkdir -p slurm/logs
VENV_PY=~/thesis_experiment/Multilingual-Refusal/venv/bin/python3
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Model: $MODEL_ALIAS  Lang: $LANG  Suffix: $SUFFIX  Start: $(date)"

"$VENV_PY" scripts/19_taxonomy_robustness.py \
    --model_alias "$MODEL_ALIAS" \
    --lang         "$LANG" \
    --suffix       "$SUFFIX"

echo "Done: $(date)"
