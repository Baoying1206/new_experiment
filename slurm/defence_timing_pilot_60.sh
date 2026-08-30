#!/bin/bash
#SBATCH --job-name=jb-defence-timing60
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/defence_timing60_%j.out

# Exp3 defence timing pilot -- 40_defence_generation_driver.py --phase timing-pilot.
# 60 real validation prompts (5 harmful + 5 benign validation_ids x 6 V2
# templates), 3 conditions (no_hook / hook_alpha_zero / global_alpha_one),
# 180 target generations total, plus a WildGuard judge pass on all 180.
# NOT a full validation sweep -- only 'timing-pilot' phase is authorized to
# run; 'validation' phase raises NotImplementedError if invoked from this
# entry point on purpose.

cd ~/new_experiment
mkdir -p slurm/logs
VENV_PY=~/thesis_experiment/Multilingual-Refusal/venv/bin/python3
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Start: $(date)"

"$VENV_PY" scripts/40_defence_generation_driver.py \
    --phase timing-pilot \
    --model_idx 1 \
    --output_path output

echo "Done: $(date)"
