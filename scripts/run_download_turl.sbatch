#!/bin/bash
# Download the full TURL CTA data (train/dev/test) from the HELM mirror on Hugging Face (CC BY 4.0),
# convert it to RAG4CTA's CSV format and make it visible as data/TURL-Wiki-CTA.
# The official TURL link (OneDrive) no longer works, see https://github.com/sunlab-osu/TURL#data
# Submit from the RACOON repo root:  mkdir -p log && sbatch scripts/run_download_turl.sbatch
#SBATCH --job-name=turl_dl
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=log/download_turl_%j.out

set -e
RAG4CTA_ROOT=${RAG4CTA_ROOT:-$HOME/RAG4CTA}
DATA=$RAG4CTA_ROOT/data
SRC=$DATA/WikiTables-TURL-CTA
BASE=https://huggingface.co/datasets/stanford-crfm/helm-scenarios/resolve/main/turl-column-type-annotation

echo "host=$(hostname) date=$(date) RAG4CTA_ROOT=$RAG4CTA_ROOT"
mkdir -p "$SRC"
for f in type_vocab.txt dev.table_col_type.json test.table_col_type.json train.table_col_type.json; do
  echo "downloading $f"
  curl -L -C - -sS --fail -o "$SRC/$f" "$BASE/$f"   # -C - resumes a partial file
  ls -la "$SRC/$f"
done

echo "converting to RAG4CTA CSV format"
python3 "$RAG4CTA_ROOT/scripts/preprocessing/convert_turl.py" --data_dir "$DATA"

# the loader (load_turl_wiki) reads data/TURL-Wiki-CTA
[ -e "$DATA/TURL-Wiki-CTA" ] || ln -s WikiTables-TURL-CTA "$DATA/TURL-Wiki-CTA"
ls -la "$SRC" "$DATA/TURL-Wiki-CTA"
echo "finished $(date)"
