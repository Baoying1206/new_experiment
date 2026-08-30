#!/bin/bash
#SBATCH --job-name=jb-defence-val
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/defence_val_%j.out

# Exp3 validation phase -- one independent, resumable job per (model, method)
# (or per (model, no_defence_target) for the two No-defence sub-jobs).
# Reads ONLY harmful validation_ids (72) + benign_validation_80 + direction_ids
# (for direction construction) + the frozen Adaptive grouping -- never
# test_ids or benign_test_100.
#
# Submit one job per combination:
#   sbatch --export=MODEL_IDX=0,METHOD=placebo   slurm/defence_validation.sh
#   sbatch --export=MODEL_IDX=0,METHOD=global    slurm/defence_validation.sh
#   sbatch --export=MODEL_IDX=0,METHOD=fixed_wei slurm/defence_validation.sh
#   sbatch --export=MODEL_IDX=0,METHOD=adaptive  slurm/defence_validation.sh
#   ... repeat for MODEL_IDX=1,2
#   sbatch --export=MODEL_IDX=0,METHOD=no_defence,NO_DEFENCE_TARGET=benign          slurm/defence_validation.sh
#   sbatch --export=MODEL_IDX=0,METHOD=no_defence,NO_DEFENCE_TARGET=harmful_rejudge slurm/defence_validation.sh
#   ... repeat for MODEL_IDX=1,2
#
# Each job writes to its own experiment3_validation_generations_{model}_{method}.jsonl
# (or _no_defence_benign.jsonl / judgements_..._no_defence_harmful.jsonl) and is
# safe to re-submit if killed -- resume is by per-record integrity check
# (record_key present AND all required fields non-null/non-empty), not just
# line count.

cd ~/new_experiment
mkdir -p slurm/logs
VENV_PY=~/thesis_experiment/Multilingual-Refusal/venv/bin/python3
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

MODEL_IDX=${MODEL_IDX:?must set MODEL_IDX=0|1|2}
METHOD=${METHOD:?must set METHOD=placebo|global|fixed_wei|adaptive|no_defence}

EXTRA_ARGS=""
if [ "$METHOD" = "no_defence" ]; then
    NO_DEFENCE_TARGET=${NO_DEFENCE_TARGET:?must set NO_DEFENCE_TARGET=benign|harmful_rejudge when METHOD=no_defence}
    EXTRA_ARGS="--no_defence_target $NO_DEFENCE_TARGET"
fi

echo "Start: $(date)"
echo "MODEL_IDX=$MODEL_IDX METHOD=$METHOD ${NO_DEFENCE_TARGET:+NO_DEFENCE_TARGET=$NO_DEFENCE_TARGET}"

"$VENV_PY" scripts/40_defence_generation_driver.py \
    --phase validation \
    --model_idx "$MODEL_IDX" \
    --method "$METHOD" \
    $EXTRA_ARGS \
    --output_path output

echo "Done: $(date)"
