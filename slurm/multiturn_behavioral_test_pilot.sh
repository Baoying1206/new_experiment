#!/bin/bash
#SBATCH --job-name=mt-bt-pilot
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/multiturn_behavioral_test_pilot_%j.out

# Multi-turn Behavioral Test pilot run
# (EXPERIMENT_CONTEXT_MULTITURN_BEHAVIOR_PROTOCOL.md, Phase A). 30
# direction_ids (the SAME 30 used by Experiment 2's own pilot, scripts/52,
# AND the single-turn Behavioral Test's own pilot, scripts/56) x 16 real
# multi-turn (user/assistant/user) context conditions,
# Meta-Llama-3.1-8B-Instruct only = 480 generations + 480 WildGuard judge
# calls. result_status=MULTITURN_PILOT_NON_RESULT throughout -- NEVER a
# behavioral finding. Purpose: verify the multi-turn generation/judge
# pipeline end to end and get real wall-clock numbers before any formal
# 3,456-generation run is even considered.
#
# NOT AUTHORIZED FOR SUBMISSION THIS ROUND. This round's authorization
# covers only: read-only audit, driver implementation, CPU dry-run/tests,
# and this slurm script itself -- NOT a real GPU run. Do not `sbatch` this
# file until a future round explicitly authorizes the real pilot.
#
# To dry-run first (CPU only, no model weights, no GPU, no WildGuard --
# this round's only authorized real execution mode):
#   srun --pty bash
#   . ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
#   cd ~/new_experiment
#   python3 scripts/60_multiturn_behavioral_test_driver.py \
#       --phase pilot --dry_run
#   python3 scripts/audits/audit_multiturn_behavioral_test_dry_run.py
#
# To submit the real pilot (FUTURE round only, after explicit
# authorization -- the driver itself will also refuse to run without
# --confirm_real_generation, as a second, independent safeguard):
#   sbatch slurm/multiturn_behavioral_test_pilot.sh

cd ~/new_experiment
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Multi-turn Behavioral Test PILOT  Start: $(date)"

python3 scripts/60_multiturn_behavioral_test_driver.py \
    --phase pilot \
    --output_dir output \
    --confirm_real_generation

echo "Done: $(date)"
