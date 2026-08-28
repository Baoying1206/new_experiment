#!/bin/bash
#SBATCH --job-name=jb-dual-axis
#SBATCH --partition=cpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/dual_axis_%j.out

# CPU-only Exp2 diagnosis (27_dual_axis_diagnosis.py). Requires all 3
# models' delta_r_h_{lang}{suffix}_{ids_key}.pt (25_extract_delta_r_h.py)
# and their v2/v3 reference directions to already exist.

cd ~/new_experiment
mkdir -p slurm/logs
VENV_PY=~/thesis_experiment/Multilingual-Refusal/venv/bin/python3
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Start: $(date)"

"$VENV_PY" scripts/27_dual_axis_diagnosis.py \
    --output_dir output \
    --lang     en \
    --suffix   _full572 \
    --ids_key  direction_ids

echo "Done: $(date)"
