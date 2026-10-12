# -*- coding: utf-8 -*-
"""
Scan ALL tables (train + valid + test) of GitTablesSC and GitTablesDB with ReFinED and write
every distinct Wikidata QID that was linked to qid_list.json (next to this file).

Format: {qid: [table_id, col_idx, cell_text]}  (one example mention per QID)

Env vars:
    RAG4CTA_ROOT       RAG4CTA checkout that holds data/GitTablesSC and data/GitTablesDB (default ~/RAG4CTA)
    REFINED_CACHE_DIR  where ReFinED keeps its model files (downloaded on first use)
Usage:  python -u build_qid_list.py [--max_tables N]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_API_KEY", "unused")  # utils.py builds an LLM client at import; the scan never calls it

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
RAG4CTA_ROOT = Path(os.getenv("RAG4CTA_ROOT", os.path.expanduser("~/RAG4CTA")))
OUT = SRC / "qid_list.json"

from KG_Linker import get_column_wise_spans  # noqa: E402
from utils import process_EL_res  # noqa: E402


def load_all_tables(subdir: str) -> list[dict]:
    """table_raw for every table across train+valid+test, one entry per table_id."""
    sys.path.insert(0, str(RAG4CTA_ROOT))
    from src.data_loader import _load_data_format
    by_table: dict = {}
    for split in ["train", "valid", "test"]:
        for r in _load_data_format(RAG4CTA_ROOT / "data", subdir, split):
            by_table.setdefault(r["table_id"], {}).setdefault(r["col_id"], r["sample_cells"])
    tables = []
    for tid, cols in by_table.items():
        columns = [cols[c] for c in sorted(cols)]
        n_rows = max(1, min(10, max((len(c) for c in columns), default=0)))
        table_raw = [[(columns[c][row] if row < len(columns[c]) else "") for c in range(len(columns))]
                     for row in range(n_rows)]
        tables.append({"table_id": tid, "table_raw": table_raw})
    return tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_tables", type=int, default=None, help="limit tables per dataset (smoke test)")
    args = ap.parse_args()

    all_qids: dict = {}
    for name, subdir in [("sc", "GitTablesSC"), ("db", "GitTablesDB")]:
        tables = load_all_tables(subdir)
        if args.max_tables:
            tables = tables[: args.max_tables]
        print(f"[{name}] {len(tables)} tables (train+valid+test)", flush=True)
        t0 = time.time()
        for i, t in enumerate(tables):
            table_cols = list(zip(*t["table_raw"]))
            spans = get_column_wise_spans("col", t["table_raw"], table_cols)
            for col_idx, col_qids in enumerate(process_EL_res(s) for s in spans):
                for cell_idx, qid in enumerate(col_qids):
                    if qid:
                        all_qids.setdefault(qid, [t["table_id"], col_idx, table_cols[col_idx][cell_idx]])
            if (i + 1) % 500 == 0:
                print(f"[{name}] {i + 1}/{len(tables)} tables, {len(all_qids)} distinct QIDs, "
                      f"{time.time() - t0:.0f}s", flush=True)
        print(f"[{name}] done: {len(tables)} tables, {len(all_qids)} distinct QIDs so far, "
              f"{time.time() - t0:.0f}s", flush=True)
        with open(OUT, "w", encoding="utf-8") as f:  # save after each dataset
            json.dump(all_qids, f, ensure_ascii=False)

    print(f"saved {len(all_qids)} distinct QIDs to {OUT}", flush=True)


if __name__ == "__main__":
    main()
