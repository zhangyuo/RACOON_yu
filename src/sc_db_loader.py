# -*- coding: utf-8 -*-
"""
Tables of GitTablesSC / GitTablesDB for RACOON_CTA.py, read directly from the source fold files
    data/<GitTablesSC|GitTablesDB>/<sp|db>_semi1_cv_{0..4}.csv      (table_id, col_idx, class, class_id, data)
where the cells of a column are separated by ';'. All tables of the dataset are used (no train/valid/test split), the
cells are the real cells (no splitting into words) and only the first 10 rows are kept, like in build_qid_list.py.

Columns without a label (class_id -1) stay in the table as context, but gold_types marks them '-1', so RACOON_CTA.py
never asks about them. Gold labels are only carried along to join the predictions back for the evaluation.

Env var RAG4CTA_ROOT: checkout that holds data/ (default ~/RAG4CTA).
"""
from __future__ import annotations  # RACOON runs on Python 3.8

import csv
import os
import sys
from pathlib import Path

ROOT = Path(os.getenv("RAG4CTA_ROOT", os.path.expanduser("~/RAG4CTA")))
MAX_ROWS = 10
_DATASETS = {"sc": ("GitTablesSC", "sp"), "db": ("GitTablesDB", "db")}


def _vocab(subdir: str) -> dict[int, str]:
    vocab: dict[int, str] = {}
    for line in open(ROOT / "data" / subdir / "type_vocab.txt", encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            vocab[int(parts[0])] = parts[1].strip()
    return vocab


def load_tables(dataset: str) -> tuple[list[dict], list[str]]:
    """Returns (tables, label_set). Each table: table_id, table_raw (row-major, <=10 rows), num_col,
    col_ids (source col_idx per column), gold_types (label name per column, '-1' = unlabeled)."""
    if dataset not in _DATASETS:
        raise ValueError(f"unsupported dataset {dataset!r}, expected one of {list(_DATASETS)}")
    subdir, prefix = _DATASETS[dataset]
    vocab = _vocab(subdir)
    csv.field_size_limit(sys.maxsize)

    by_table: dict[str, dict[int, tuple[list[str], str]]] = {}
    for i in range(5):
        with open(ROOT / "data" / subdir / f"{prefix}_semi1_cv_{i}.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cells = (r["data"] or "").split(";")[:MAX_ROWS]
                class_id = int(float(r["class_id"])) if (r["class_id"] or "").strip() else -1
                gold = vocab.get(class_id, "-1") if class_id >= 0 else "-1"
                by_table.setdefault(r["table_id"], {}).setdefault(int(r["col_idx"]), (cells, gold))

    tables = []
    for table_id, cols in by_table.items():
        col_ids = sorted(cols)
        columns = [cols[c][0] for c in col_ids]
        n_rows = max(1, max((len(c) for c in columns), default=0))
        tables.append({
            "table_id": table_id,
            "table_raw": [[(columns[c][r] if r < len(columns[c]) else "") for c in range(len(columns))]
                          for r in range(n_rows)],
            "num_col": len(col_ids),
            "col_ids": col_ids,
            "gold_types": [cols[c][1] for c in col_ids],
        })
    return tables, [vocab[k] for k in sorted(vocab)]
