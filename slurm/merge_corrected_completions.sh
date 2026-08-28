#!/bin/bash
#SBATCH --job-name=jb-merge-corrected
#SBATCH --partition=cpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/merge_corrected_%j.out

# CPU-only merge of the KEPT old-taxonomy conditions (plain/placebo/
# prefix_injection/refusal_suppression/persona_roleplay/encoding_obfuscation)
# with the freshly-generated new mechanisms (payload_splitting/
# distractor_instructions) into one 8-condition corrected-taxonomy
# completions file (31_merge_corrected_completions.py). Requires both
# completions_{lang}_full572.json and completions_{lang}_full572_newmechs.json
# to already exist for this model.

MODEL_ALIASES=(
    "Qwen2.5-7B-Instruct"
    "Meta-Llama-3.1-8B-Instruct"
    "gemma-2-9b-it"
)
MODEL_IDX=${MODEL_IDX:-0}
MODEL_ALIAS=${MODEL_ALIASES[$MODEL_IDX]}

cd ~/new_experiment
mkdir -p slurm/logs
VENV_PY=~/thesis_experiment/Multilingual-Refusal/venv/bin/python3
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Model: $MODEL_ALIAS  Start: $(date)"

"$VENV_PY" scripts/31_merge_corrected_completions.py \
    --output_dir  output \
    --model_alias "$MODEL_ALIAS" \
    --lang        en \
    --old_suffix  _full572 \
    --new_suffix  _full572_newmechs \
    --corrected_suffix _full572_corrected

echo "Done: $(date)"
