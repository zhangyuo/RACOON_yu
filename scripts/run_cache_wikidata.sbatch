#!/bin/bash
# Query Wikidata for every QID of the list, then verify the cache. One job, one log: log/cache_wikidata.out
# Submit from the RACOON repo root:  sbatch scripts/run_cache_wikidata.sbatch
# TAGS=baseline (default) selects src/qid_list.baseline.json
#SBATCH --job-name=wd_cache
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=log/cache_wikidata.out

set -e
mkdir -p log
PY=${PY:-$HOME/anaconda3/envs/racoon/bin/python}
TAGS=${TAGS:-baseline}
echo "host=$(hostname) date=$(date) tags=$TAGS"
$PY -u src/cache_wikidata_batch.py --tags "$TAGS"
echo "=== verification ==="
$PY -u src/check_wikidata_cache.py --tags "$TAGS" --sample 100
echo "finished $(date)"
