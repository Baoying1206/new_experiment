#!/bin/bash
#SBATCH --job-name=bt-formal
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/behavioral_test_formal_%j.out

# Behavioral Test FORMAL run (one model per submission). Implements
# EXPERIMENT_BEHAVIORAL_TEST_PROTOCOL.md Sec 2/6. 72 validation_ids x 16
# context conditions = 1,152 generations + judge calls per model (3,456
# across all 3). result_status=BEHAVIORAL_TEST_FORMAL_RESULT.
#
# NOT AUTHORIZED to run without a separate, explicit go-ahead AFTER the
# pilot (slurm/behavioral_test_pilot.sh) has been reviewed -- protocol
# Sec 9.
#
# Submit once per model:
#   sbatch --export=MODEL_IDX=0 slurm/behavioral_test_formal.sh
#   sbatch --export=MODEL_IDX=1 slurm/behavioral_test_formal.sh
#   sbatch --export=MODEL_IDX=2 slurm/behavioral_test_formal.sh

MODEL_IDX=${MODEL_IDX:-0}

cd ~/new_experiment
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Behavioral Test FORMAL  MODEL_IDX=$MODEL_IDX  Start: $(date)"

python3 scripts/56_behavioral_test_generation_and_judge_driver.py \
    --phase formal \
    --model_idx "$MODEL_IDX" \
    --output_dir output

echo "Done: $(date)"
