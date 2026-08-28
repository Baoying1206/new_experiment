#!/bin/bash
#SBATCH --job-name=jb-gen-newmechs
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/gen_newmechs_%j.out

# Generates completions for ONLY the 2 new corrected-taxonomy mechanisms
# (payload_splitting, distractor_instructions) -- 572 x 2 = 1144 rows, not
# the full 572 x 8 = 4576 a complete regeneration would cost. Reuses
# 03_generate_and_label.py unchanged; the generation_input file was already
# built locally via 02_build_templated_data.py --only_mechanisms
# payload_splitting,distractor_instructions --skip_plain --suffix
# _full572_newmechs (committed to data/).
#
# Override via --export=MODEL_IDX=...
#   sbatch --export=MODEL_IDX=0 slurm/generate_newmechs.sh

MODEL_PATHS=(
    "/home/h24/baga0553/models/Qwen2.5-7B-Instruct"
    "/home/h24/baga0553/models/Llama-3.1-8B-Instruct"
    "/home/h24/baga0553/models/gemma-2-9b-it"
)
MODEL_ALIASES=(
    "Qwen2.5-7B-Instruct"
    "Meta-Llama-3.1-8B-Instruct"
    "gemma-2-9b-it"
)

MODEL_IDX=${MODEL_IDX:-0}
MODEL_PATH=${MODEL_PATHS[$MODEL_IDX]}
MODEL_ALIAS=${MODEL_ALIASES[$MODEL_IDX]}

cd ~/new_experiment
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Model: $MODEL_ALIAS  Start: $(date)"

python scripts/03_generate_and_label.py \
    --model_path  "$MODEL_PATH" \
    --model_alias "$MODEL_ALIAS" \
    --lang        en \
    --output_dir  output \
    --suffix      _full572_newmechs \
    --batch_size  8

echo "Done: $(date)"
