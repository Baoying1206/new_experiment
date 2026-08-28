#!/bin/bash
#SBATCH --job-name=jb-within-disp
#SBATCH --partition=cpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/within_disp_%j.out

# CPU-only within-template dispersion analysis (29_within_template_dispersion.py).
# Reads the already-existing paired_diffs_{lang}{suffix}.pt (18_extract_paired_diffs.py's
# output, already used for Exp1) -- no new GPU extraction needed.

cd ~/new_experiment
mkdir -p slurm/logs
VENV_PY=~/thesis_experiment/Multilingual-Refusal/venv/bin/python3
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Start: $(date)"

"$VENV_PY" scripts/29_within_template_dispersion.py \
    --output_dir output \
    --lang     en \
    --suffix   _full572

echo "Done: $(date)"
