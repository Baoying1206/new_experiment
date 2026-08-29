#!/bin/bash
#SBATCH --job-name=jb-llama-co-div
#SBATCH --partition=cpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/llama_co_div_%j.out

# CPU-only Exp2 (34_llama_co_divergence_diagnostic.py) -- explanatory diagnostic
# for why Meta-Llama-3.1-8B-Instruct's CO group (prefix_injection,
# refusal_suppression, persona_roleplay) shows Delta_CO < 0 at the fixed layer
# (index 19) in 33_canonical_taxonomy_geometry.py's canonical V2 result.
# Requires paired_diffs_en_full572_corrected.pt for Llama only -- reuses the
# already-committed output/canonical_v2/paired_diffs_audit.json, does not
# re-run the .pt integrity audit.
#
# Local dry run (synthetic data, real n_layers=32, small d_model) completed
# in ~6s; real d_model=4096 scales the H3/H4/D_angle bootstrap tensor ops
# linearly, expected to stay in the low minutes, well under any time limit --
# no --time set (matches this repo's convention of leaving it unset unless a
# job has previously timed out).

cd ~/new_experiment
mkdir -p slurm/logs
VENV_PY=~/thesis_experiment/Multilingual-Refusal/venv/bin/python3
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

echo "Start: $(date)"

"$VENV_PY" scripts/34_llama_co_divergence_diagnostic.py \
    --output_dir output \
    --lang       en \
    --suffix     _full572_corrected

echo "Done: $(date)"
