#!/bin/bash
#SBATCH --job-name=jb-audit-pd
#SBATCH --partition=cpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/audit_pd_%j.out

# CPU-only, read-only .pt audit (audit_paired_diffs_corrected.py). Only
# needs torch -- no `pipeline` import, no completions files touched (those
# can't verify NaN/Inf in the extracted activations anyway).

cd ~/new_experiment
mkdir -p slurm/logs
VENV_PY=~/thesis_experiment/Multilingual-Refusal/venv/bin/python3
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Start: $(date)"

"$VENV_PY" scripts/audits/audit_paired_diffs_corrected.py \
    --output_dir output \
    --lang       en \
    --suffix     _full572_corrected

echo "Done: $(date)"
