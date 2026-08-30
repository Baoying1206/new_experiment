#!/bin/bash
#SBATCH --job-name=jb-exp3-coverage
#SBATCH --partition=cpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/exp3_coverage_%j.out

# CPU-only Exp3 common-direction coverage audit (35_common_direction_coverage_audit.py).
# Read-only against the existing paired_diffs_{lang}_full572_corrected.pt for all 3
# models -- no GPU, no new completions, no steering/defence, no changes to
# scripts 33/34 or their outputs. Only direction_ids (300) are read;
# validation_ids/test_ids are never touched by this script.
#
# Local dry run (synthetic data, real n_layers per model, small d_model)
# completed in ~8s for all 3 models combined. Real d_model (3584-4096)
# scales the bootstrap (2000 reps x 7 mechanism aggregations x 2
# g-definitions, at each model's fixed layer only) roughly in line with
# script 34's H4 cost; expected total runtime across all 3 models is in the
# 10-20 minute range, unlikely to exceed 30 minutes -- no --time limit set
# (matches this repo's convention of leaving it unset unless a job has
# previously timed out).

cd ~/new_experiment
mkdir -p slurm/logs
VENV_PY=~/thesis_experiment/Multilingual-Refusal/venv/bin/python3
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Start: $(date)"

"$VENV_PY" scripts/35_common_direction_coverage_audit.py \
    --output_dir output \
    --lang       en \
    --suffix     _full572_corrected

echo "Done: $(date)"
