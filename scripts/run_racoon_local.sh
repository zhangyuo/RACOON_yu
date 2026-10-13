#!/bin/bash
# Run RACOON_CTA.py on a cs-gpu machine with the environment, models and data on NFS (no SLURM).
# Usage (background, with a log):
#   GPU=1 DATASET=sc MAX_TABLES=20 nohup bash scripts/run_racoon_local.sh > log/racoon_sc.out 2>&1 &
# GPU: CUDA device index | DATASET: sc | db | turl | MAX_TABLES: omit for all tables
# The LLM endpoint comes from the VLLM_* lines of $RAG4CTA_ROOT/.env.
set -e
cd "$(dirname "$0")/.."
NFS=${NFS:-/nfs/srv/data2/af62lp@cs.aau.dk}
export RAG4CTA_ROOT=${RAG4CTA_ROOT:-$NFS/RAG4CTA}
export REFINED_CACHE_DIR=${REFINED_CACHE_DIR:-$NFS/refined_cache}
PY=${PY:-$NFS/miniconda3/envs/racoon/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-1}
DATASET=${DATASET:-sc}

set -a; source <(grep -E '^VLLM_[A-Z_]+=' "$RAG4CTA_ROOT/.env"); set +a
echo "host=$(hostname) date=$(date) dataset=$DATASET max_tables=${MAX_TABLES:-all} gpu=$CUDA_VISIBLE_DEVICES llm=$VLLM_MODEL endpoint=$VLLM_BASE_URL"
echo "rows per column: 6 | context: col | info: type"
cd src/CTA
$PY -u RACOON_CTA.py --dataset "$DATASET" --context col --info type ${MAX_TABLES:+--max_tables $MAX_TABLES}
echo "finished $(date)"
