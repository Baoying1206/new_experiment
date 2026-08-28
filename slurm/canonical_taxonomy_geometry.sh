#!/bin/bash
#SBATCH --job-name=jb-canon-geo
#SBATCH --partition=cpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/canon_geo_%j.out

# CPU-only Exp1 canonical V2 taxonomy geometry (33_canonical_taxonomy_geometry.py).
# Requires paired_diffs_{lang}_full572_corrected.pt for all 3 models (18_extract_paired_diffs.py
# --mechanisms matching the V2 active_mechanisms) -- run
# audit_paired_diffs_corrected.sh first and check its output is clean.
#
# 2000-rep bootstrap x 6 (estimator x raw/pc) combinations x 3 models --
# dry-run timing extrapolation suggests this could take several minutes to
# ~15 minutes with real d_model; no --time limit set (matches this repo's
# convention of leaving it unset unless a job has previously timed out).

cd ~/new_experiment
mkdir -p slurm/logs
VENV_PY=~/thesis_experiment/Multilingual-Refusal/venv/bin/python3
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Start: $(date)"

"$VENV_PY" scripts/33_canonical_taxonomy_geometry.py \
    --output_dir output \
    --lang       en \
    --suffix     _full572_corrected

echo "Done: $(date)"
