#!/bin/bash
#SBATCH --job-name=jb-pairwise-cos
#SBATCH --partition=cpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/pairwise_cos_%j.out

# CPU-only pairwise template cosine matrix (30_pairwise_template_cosine.py).
# Reads the already-existing paired_diffs_{lang}{suffix}.pt -- no new GPU
# extraction needed.
#
# Override via --export=SUFFIX=...,TAXONOMY=old|corrected
#   Corrected taxonomy: sbatch --export=SUFFIX=_full572_corrected,TAXONOMY=corrected slurm/pairwise_template_cosine.sh

SUFFIX=${SUFFIX:-_full572}
TAXONOMY=${TAXONOMY:-old}

cd ~/new_experiment
mkdir -p slurm/logs
VENV_PY=~/thesis_experiment/Multilingual-Refusal/venv/bin/python3
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Suffix: $SUFFIX  Taxonomy: $TAXONOMY  Start: $(date)"

"$VENV_PY" scripts/30_pairwise_template_cosine.py \
    --output_dir output \
    --lang       en \
    --suffix     "$SUFFIX" \
    --taxonomy   "$TAXONOMY"

echo "Done: $(date)"
