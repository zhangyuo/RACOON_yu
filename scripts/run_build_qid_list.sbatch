#!/bin/bash
# Build qid_list.json (ReFinED scan of GitTablesSC + GitTablesDB, all splits) on a SLURM GPU node.
# Submit from the RACOON repo root:  sbatch scripts/run_build_qid_list.sbatch
# Does not pin a GPU type: SLURM gives whichever GPU is free.
#SBATCH --job-name=qid_list
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=log/build_qid_list_%j.out

set -e
mkdir -p log
export RAG4CTA_ROOT=${RAG4CTA_ROOT:-$HOME/RAG4CTA}
export REFINED_CACHE_DIR=${REFINED_CACHE_DIR:-$HOME/refined_cache}
PY=${PY:-$HOME/anaconda3/envs/racoon/bin/python}

echo "host=$(hostname) date=$(date) PY=$PY RAG4CTA_ROOT=$RAG4CTA_ROOT REFINED_CACHE_DIR=$REFINED_CACHE_DIR"
nvidia-smi -L || true
$PY -u src/build_qid_list.py ${MAX_TABLES:+--max_tables $MAX_TABLES}
echo "finished $(date)"
