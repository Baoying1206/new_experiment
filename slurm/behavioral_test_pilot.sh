#!/bin/bash
#SBATCH --job-name=bt-pilot
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/behavioral_test_pilot_%j.out

# Behavioral Test pilot run (EXPERIMENT_BEHAVIORAL_TEST_PROTOCOL.md Sec 6).
# 30 direction_ids (the SAME 30 used by Experiment 2's own pilot,
# scripts/52) x 16 context conditions, Meta-Llama-3.1-8B-Instruct only =
# 480 generations + 480 WildGuard judge calls. result_status=PILOT_NON_RESULT.
# NOT a behavioral finding -- purpose is to verify the generation/judge
# pipeline and get real wall-clock numbers for formal-scale GPU planning.
#
# To dry-run first (CPU only, no model weights, no GPU, no WildGuard):
#   srun --pty bash
#   . ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
#   cd ~/new_experiment
#   python3 scripts/56_behavioral_test_generation_and_judge_driver.py \
#       --phase pilot --dry_run
#
# To submit the real pilot:
#   sbatch slurm/behavioral_test_pilot.sh

cd ~/new_experiment
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Behavioral Test PILOT  Start: $(date)"

python3 scripts/56_behavioral_test_generation_and_judge_driver.py \
    --phase pilot \
    --output_dir output

echo "Done: $(date)"
