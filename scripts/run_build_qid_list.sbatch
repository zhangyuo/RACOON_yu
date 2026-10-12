#!/bin/bash
# Build qid_list.json (ReFinED scan of GitTablesSC, GitTablesDB and TURL, all splits) on a SLURM GPU node.
# Submit from the RACOON repo root:  sbatch scripts/run_build_qid_list.sbatch
# Does not pin a GPU type: SLURM gives whichever GPU is free.
# Smoke test (5 tables per unit, separate result files):  MAX_TABLES=5 sbatch scripts/run_build_qid_list.sbatch
# Resumable: re-submitting continues after the last finished unit.
#SBATCH --job-name=qid_list
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=log/build_qid_list.out

set -e
mkdir -p log
export RAG4CTA_ROOT=${RAG4CTA_ROOT:-$HOME/RAG4CTA}
export REFINED_CACHE_DIR=${REFINED_CACHE_DIR:-$HOME/refined_cache}
PY=${PY:-$HOME/anaconda3/envs/racoon/bin/python}

echo "host=$(hostname) date=$(date) PY=$PY RAG4CTA_ROOT=$RAG4CTA_ROOT REFINED_CACHE_DIR=$REFINED_CACHE_DIR"
nvidia-smi -L || true

# ReFinED memory-maps ~35GB of entity arrays. Random first-touch reads on the shared filesystem cost ~5s per table,
# so read the files once, sequentially and in parallel, to get them into the page cache before scanning.
t0=$(date +%s)
for f in "$REFINED_CACHE_DIR"/wikipedia_model/*.np "$REFINED_CACHE_DIR"/wikidata_data/*.np "$REFINED_CACHE_DIR"/wikidata_data/*.lmdb; do
  cat "$f" > /dev/null &
done
wait
echo "page cache warm-up: $(( $(date +%s) - t0 ))s"
$PY -u src/build_qid_list.py ${UNITS:+--units $UNITS} ${MAX_TABLES:+--max_tables $MAX_TABLES}
echo "finished $(date)"
