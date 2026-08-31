#!/bin/bash
#SBATCH --job-name=jb-defence-test
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/defence_test_%j.out

# Exp3 TEST phase (one-shot, held-out) -- one independent, resumable job per
# (model, method), or per (model, no_defence_target) for the two No-defence
# sub-jobs. method in {no_defence, fixed_wei, adaptive} only -- global/
# placebo are not test-phase methods (49_defence_test_driver.py's --method
# choices reject them at argparse level). fixed_wei/adaptive take NO --alpha
# here -- 49 reads the single frozen alpha from
# experiment3_defence_frozen_config.json and hard-stops if the runtime
# direction/generation_config_hash, fixed layer, or live Adaptive grouping
# disagree with what was frozen.
#
# Submit one job per combination. STAGE defaults to "generate"; run again
# with STAGE=judge AFTER generation finishes for that (model, method) --
# generation and judging are two separate jobs, except no_defence/
# harmful_rejudge which always does both in one step (STAGE is ignored for
# it):
#   sbatch --export=MODEL_IDX=0,METHOD=fixed_wei slurm/defence_test.sh
#   sbatch --export=MODEL_IDX=0,METHOD=fixed_wei,STAGE=judge slurm/defence_test.sh
#   sbatch --export=MODEL_IDX=0,METHOD=adaptive  slurm/defence_test.sh
#   sbatch --export=MODEL_IDX=0,METHOD=adaptive,STAGE=judge slurm/defence_test.sh
#   ... repeat for MODEL_IDX=1,2
#   sbatch --export=MODEL_IDX=0,METHOD=no_defence,NO_DEFENCE_TARGET=benign          slurm/defence_test.sh
#   sbatch --export=MODEL_IDX=0,METHOD=no_defence,NO_DEFENCE_TARGET=benign,STAGE=judge slurm/defence_test.sh
#   sbatch --export=MODEL_IDX=0,METHOD=no_defence,NO_DEFENCE_TARGET=harmful_rejudge slurm/defence_test.sh
#   ... repeat for MODEL_IDX=1,2
#
# Each job writes to its own experiment3_test_generations_{model}_{method}.jsonl
# (or _no_defence_benign.jsonl / judgements_..._no_defence_harmful.jsonl) and
# is safe to re-submit if killed -- resume is by per-record integrity check
# (record_key present AND all required fields non-null/non-empty), reusing
# 40_defence_generation_driver.py's already-proven record_is_valid.

cd ~/new_experiment
mkdir -p slurm/logs
VENV_PY=~/thesis_experiment/Multilingual-Refusal/venv/bin/python3
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

MODEL_IDX=${MODEL_IDX:?must set MODEL_IDX=0|1|2}
METHOD=${METHOD:?must set METHOD=no_defence|fixed_wei|adaptive}
STAGE=${STAGE:-generate}

EXTRA_ARGS=""
if [ "$METHOD" = "no_defence" ]; then
    NO_DEFENCE_TARGET=${NO_DEFENCE_TARGET:?must set NO_DEFENCE_TARGET=benign|harmful_rejudge when METHOD=no_defence}
    EXTRA_ARGS="--no_defence_target $NO_DEFENCE_TARGET"
fi

echo "Start: $(date)"
echo "MODEL_IDX=$MODEL_IDX METHOD=$METHOD STAGE=$STAGE ${NO_DEFENCE_TARGET:+NO_DEFENCE_TARGET=$NO_DEFENCE_TARGET}"

"$VENV_PY" scripts/49_defence_test_driver.py \
    --model_idx "$MODEL_IDX" \
    --method "$METHOD" \
    --stage "$STAGE" \
    $EXTRA_ARGS \
    --output_path output

echo "Done: $(date)"
