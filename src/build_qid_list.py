# -*- coding: utf-8 -*-
"""
Scan ALL tables of GitTablesSC, GitTablesDB and TURL (train + valid/dev + test) with ReFinED and
write every distinct Wikidata QID that was linked to qid_list.json (next to this file).

Format: {qid: [table_id, col_idx, cell_text]}  (one example mention per QID)

The scan feeds ReFinED exactly what RACOON_CTA.py feeds it: the first 10 rows of a table, linked column by column.
  * SC / DB  : the source fold files data/GitTablesSC/sp_semi1_cv_{0..4}.csv and data/GitTablesDB/db_semi1_cv_{0..4}.csv,
               cells split on ';' (all folds, no train/valid/test split).
  * TURL     : the original *.table_col_type.json parsed with RACOON's own parse_example
               (data/WikiTables-TURL-CTA/{test,dev,train}.table_col_type.json).
  The converted train/valid/test CSVs are not used: they split multi-word cells into words.

Units are scanned in this order: sc, db, turl_test, turl_dev, turl_train. qid_list.json is saved after every unit
(and every 20000 tables inside a unit); finished units are recorded in qid_list.progress.json, so a re-run resumes.

Env vars:
    RAG4CTA_ROOT       RAG4CTA checkout that holds the data/ directory (default ~/RAG4CTA)
    REFINED_CACHE_DIR  where ReFinED keeps its model files (downloaded on first use)
Usage:  python -u build_qid_list.py [--units sc,db,turl_test,turl_dev,turl_train] [--max_tables N]
"""
import argparse
import csv
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
PROGRESS = SRC / "qid_list.progress.json"
SAVE_EVERY = 20000
MAX_ROWS = 10  # RACOON_CTA.py links only the first 10 rows

from KG_Linker import get_column_wise_spans  # noqa: E402
from utils import process_EL_res, parse_example  # noqa: E402

UNITS = ["sc", "db", "turl_test", "turl_dev", "turl_train"]
TURL_FILES = {"turl_test": "test", "turl_dev": "dev", "turl_train": "train"}


def save(all_qids, done):
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(all_qids, f, ensure_ascii=False)
    with open(PROGRESS, "w", encoding="utf-8") as f:
        json.dump(done, f)


def scan_table(table_id, rows, all_qids):
    """rows: list of rows (cells as str or None). Links the first MAX_ROWS rows column by column, like RACOON_CTA.py."""
    if not rows or not rows[0]:
        return
    x, y = min(len(rows), MAX_ROWS), len(rows[0])
    table_cols = [["" if rows[r][c] is None else rows[r][c] for r in range(x)] for c in range(y)]
    table_row = list(zip(*table_cols))
    spans = get_column_wise_spans("col", table_row, table_cols)
    for col_idx, col_qids in enumerate(process_EL_res(s) for s in spans):
        col_cells = table_cols[col_idx] if col_idx < len(table_cols) else []
        for cell_idx, qid in enumerate(col_qids):
            if qid:
                # the linker groups mentions by splitting the column text on "||"; a cell that itself contains "||"
                # shifts the groups, so the example cell text is only a hint (the QID is what matters)
                all_qids.setdefault(qid, [table_id, col_idx, col_cells[cell_idx] if cell_idx < len(col_cells) else ""])


def gittables_tables(subdir, prefix):
    """(table_id, rows) for every table of a GitTables dataset, read from the source fold files
    data/<subdir>/<prefix>_semi1_cv_{0..4}.csv (columns table_id, col_idx, class, class_id, data;
    the cells of a column are separated by ';'). No train/valid/test split, no re-conversion."""
    csv.field_size_limit(sys.maxsize)
    by_table: dict = {}
    for i in range(5):
        with open(RAG4CTA_ROOT / "data" / subdir / f"{prefix}_semi1_cv_{i}.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cells = (r["data"] or "").split(";")
                by_table.setdefault(r["table_id"], {}).setdefault(int(r["col_idx"]), cells)
    for tid, cols in by_table.items():
        columns = [cols[c][:MAX_ROWS] for c in sorted(cols)]
        n_rows = max(1, max((len(c) for c in columns), default=0))
        yield tid, [[(columns[c][r] if r < len(columns[c]) else "") for c in range(len(columns))] for r in range(n_rows)]


def turl_tables(split):
    path = RAG4CTA_ROOT / "data" / "WikiTables-TURL-CTA" / f"{split}.table_col_type.json"
    with open(path, "r") as f:
        examples = json.load(f)
    for example in examples:
        table_raw = parse_example(example)[0]
        yield f"turl:{example[0]}", table_raw.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default=",".join(UNITS))
    ap.add_argument("--max_tables", type=int, default=None, help="limit tables per unit (smoke test)")
    args = ap.parse_args()

    # a smoke test must not touch the real result files
    global OUT, PROGRESS
    if args.max_tables:
        OUT, PROGRESS = SRC / "qid_list.smoke.json", SRC / "qid_list.smoke.progress.json"

    all_qids = json.load(open(OUT)) if OUT.exists() else {}
    done = json.load(open(PROGRESS)) if PROGRESS.exists() else []
    print(f"resuming with {len(all_qids)} QIDs, finished units: {done}", flush=True)

    for unit in args.units.split(","):
        if unit in done:
            print(f"[{unit}] already done, skipping", flush=True)
            continue
        if unit in TURL_FILES:
            tables = turl_tables(TURL_FILES[unit])
        else:
            tables = gittables_tables(*{"sc": ("GitTablesSC", "sp"), "db": ("GitTablesDB", "db")}[unit])
        t0, n, failed = time.time(), 0, []
        for table_id, rows in tables:
            try:
                scan_table(table_id, rows, all_qids)
            except Exception as e:  # one odd table must not stop a run that takes hours
                failed.append(table_id)
                if len(failed) <= 5:
                    print(f"[{unit}] skipped table {table_id}: {type(e).__name__}: {e}", flush=True)
            n += 1
            if n % 500 == 0:
                print(f"[{unit}] {n} tables, {len(all_qids)} distinct QIDs, {time.time() - t0:.0f}s", flush=True)
            if n % SAVE_EVERY == 0:
                save(all_qids, done)
            if args.max_tables and n >= args.max_tables:
                break
        done.append(unit)
        save(all_qids, done)
        print(f"[{unit}] done: {n} tables ({len(failed)} skipped after an error), {len(all_qids)} distinct QIDs so far, "
              f"{time.time() - t0:.0f}s", flush=True)

    print(f"saved {len(all_qids)} distinct QIDs to {OUT}", flush=True)


if __name__ == "__main__":
    main()
