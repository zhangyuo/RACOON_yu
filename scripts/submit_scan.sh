#!/bin/bash
# Submit the QID scan as parallel SLURM jobs (one GPU each):
#   qid_main    : sc, db, turl_test, turl_dev           -> src/qid_list.main.json
#   qid_train0..: shards of turl_train (400k tables)    -> src/qid_list.train<k>.json
# Run from the RACOON repo root:  bash scripts/submit_scan.sh [TRAIN_SHARDS=8]
# Afterwards (or before caching) the files are merged by:  python src/merge_qid_lists.py
N=${1:-8}
mkdir -p log
UNITS=sc,db,turl_test,turl_dev TAG=main sbatch --job-name=qid_main -o log/qid_main.out scripts/run_build_qid_list.sbatch
for k in $(seq 0 $((N - 1))); do
  UNITS=turl_train SHARD=$k/$N TAG=train$k sbatch --job-name=qid_train$k -o log/qid_train$k.out scripts/run_build_qid_list.sbatch
done
