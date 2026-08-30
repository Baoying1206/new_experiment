#!/bin/bash
#SBATCH --job-name=jb-defence-pilot
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/defence_pilot_%j.out

# Minimal REAL-GPU pilot for the Exp3 defence protocol (39_defence_pilot.py).
# NOT the full driver -- 4 real validation_ids instructions, Llama only,
# persona_roleplay template, Global direction, alpha=1.0. Purpose: confirm
# the hook fires exactly once per batch on a real model, get a real GPU
# name (sacct/sinfo gave nothing on this cluster), and get isolated
# target-gen-only and WildGuard-only throughput numbers (historical logs
# fuse both into one timestamp). No result from this run is meant to be
# reported as a defence-efficacy finding.

cd ~/new_experiment
mkdir -p slurm/logs
VENV_PY=~/thesis_experiment/Multilingual-Refusal/venv/bin/python3
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Start: $(date)"

"$VENV_PY" scripts/39_defence_pilot.py \
    --model_path /home/h24/baga0553/models/Llama-3.1-8B-Instruct \
    --output_dir output

echo "Done: $(date)"
