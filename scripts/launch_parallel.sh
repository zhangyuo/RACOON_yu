#!/bin/bash
# Start RACOON_CTA.py runs in parallel on this machine, one process per job, spread over GPUs.
# Usage:  GPUS="1 2 3" bash scripts/launch_parallel.sh <stage>      stage = small | turl
#   small: BiodivTab-DB/Wiki (both settings), SC and DB (single_label)
#   db   : only the two DB shards of the small stage
#   turl : TURL, both settings, TURL_SHARDS shards each (default 12), label space LABEL_SPACE (default freebase)
# One log per job in log/, all PIDs in log/pids.txt. Stop a job by PID (never with pkill -f).
cd "$(dirname "$0")/.."
GPUS=(${GPUS:-1 2 3})
STAGE=${1:?stage: small | turl}
mkdir -p log
n=0
launch() {  # dataset setting shard(optional)
  local ds=$1 st=$2 sh=$3 g=${GPUS[$((n % ${#GPUS[@]}))]}
  local name="${ds}${LABEL_SPACE:+_$LABEL_SPACE}_${st}${sh:+_shard${sh/\//of}}"
  env GPU=$g DATASET=$ds LABEL_SETTING=$st ${LABEL_SPACE:+LABEL_SPACE=$LABEL_SPACE} ${sh:+SHARD=$sh} nohup bash scripts/run_racoon_local.sh > "log/run_${name}.out" 2>&1 &
  echo "$! $name gpu$g $(date +%H:%M:%S)" | tee -a log/pids.txt
  n=$((n + 1))
}
case "$STAGE" in
  small)
    for ds in biodiv_db biodiv_wiki; do for st in single_label multi_label; do launch $ds $st; done; done
    launch sc single_label
    launch db single_label 0/2
    launch db single_label 1/2 ;;
  db)
    launch db single_label 0/2
    launch db single_label 1/2 ;;
  turl)
    N=${TURL_SHARDS:-12}   # TURL with the 255 Freebase names (default label space), N shards per setting
    for st in single_label multi_label; do for k in $(seq 0 $((N - 1))); do LABEL_SPACE=${LABEL_SPACE:-freebase} launch turl $st $k/$N; done; done ;;
esac
