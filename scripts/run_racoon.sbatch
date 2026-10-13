#!/bin/bash
# Run RACOON_CTA.py on one dataset. Submit from the RACOON repo root:
#   DATASET=sc MAX_TABLES=20 sbatch scripts/run_racoon.sbatch     # smoke test
#   DATASET=sc sbatch scripts/run_racoon.sbatch                   # full run (sc | db | turl)
# The LLM endpoint comes from the VLLM_* lines of $RAG4CTA_ROOT/.env (VLLM_API_KEY, VLLM_BASE_URL, VLLM_MODEL).
# Output: outputs/<model>/<dataset>/RACOON_col_type.csv  (turl: outputs/<model>/RACOON_col_type.csv)
#SBATCH --job-name=racoon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=log/racoon.out

set -e
mkdir -p log
export RAG4CTA_ROOT=${RAG4CTA_ROOT:-$HOME/RAG4CTA}
export REFINED_CACHE_DIR=${REFINED_CACHE_DIR:-$HOME/refined_cache}
PY=${PY:-$HOME/anaconda3/envs/racoon/bin/python}
DATASET=${DATASET:-sc}

set -a; source <(grep -E '^VLLM_[A-Z_]+=' "$RAG4CTA_ROOT/.env"); set +a
echo "host=$(hostname) date=$(date) dataset=$DATASET max_tables=${MAX_TABLES:-all} llm=${VLLM_MODEL:-default} endpoint=${VLLM_BASE_URL:-default}"
nvidia-smi -L || true

# read ReFinED's big entity arrays into the page cache first (see run_build_qid_list.sbatch)
for f in "$REFINED_CACHE_DIR"/wikipedia_model/*.np "$REFINED_CACHE_DIR"/wikidata_data/*.np "$REFINED_CACHE_DIR"/wikidata_data/*.lmdb; do
  cat "$f" > /dev/null &
done
wait

cd src/CTA
$PY -u RACOON_CTA.py --dataset "$DATASET" --context col --info type ${MAX_TABLES:+--max_tables $MAX_TABLES}
echo "finished $(date)"
