# -*- coding: utf-8 -*-
"""
Tables of GitTablesSC / GitTablesDB / BiodivTab-DB-CTA / BiodivTab-Wiki-CTA / SOTAB-CTA for RACOON_CTA.py, read from the shared dataset files
    data/<dataset>/{train,valid,test}.csv                       (table_id, column_index, label, data; a column with several gold
                                                                 types has one row per gold type, every cell of the table is here)
    data/<dataset>/single_label/{split}.csv                     columns with exactly one gold type
    data/<dataset>/multi_label/{split}.csv                      columns with two or more gold types, `label` = ids joined by ';'
                                                                 in gold order (first = primary)
    data/<dataset>/type_vocab.txt                               (id <TAB> label name)
The table content comes from the top-level file, the gold types of a column from the two setting folders (the union is every
labeled column once). A column with several gold types gets gold_types[c] = "name1;name2" in gold order, gold_labels[c] = [..].
The split is by whole table, so every table sits completely in one file. The cells of a column are separated by ';',
they are the real cells (no splitting into words) and only the first 6 rows are kept, like build_qid_list.py.

Columns without a label (label -1) stay in the table as context, but gold_types marks them '-1', so RACOON_CTA.py
never asks about them. Gold labels are only carried along to join the predictions back for the evaluation.

SOTAB-CTA: the shared file joins the cells of a column with a plain space, so the real cells come from the official SOTAB tables
(sotab_cells.py, first 6 non-empty cells of every column); structure (all columns) and gold are read from the shared files as above.

Env var RAG4CTA_ROOT: checkout that holds data/ (default ~/RAG4CTA).
"""
from __future__ import annotations  # RACOON runs on Python 3.8

import csv
import os
import sys
from pathlib import Path

ROOT = Path(os.getenv("RAG4CTA_ROOT", os.path.expanduser("~/RAG4CTA")))
MAX_ROWS = 6  # project rule: only the first 6 rows of every column are used (RACOON default would be 10)
_DATASETS = {"sc": "GitTablesSC", "db": "GitTablesDB", "biodiv_db": "BiodivTab-DB-CTA", "biodiv_wiki": "BiodivTab-Wiki-CTA",
             "sotab": "SOTAB-CTA", "turl": "TURL-Wiki-CTA"}
_SPLITS = ("train", "valid", "test")


def _vocab(base: Path) -> dict[int, str]:
    vocab: dict[int, str] = {}
    for line in open(base / "type_vocab.txt", encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            vocab[int(parts[0])] = parts[1].strip()
    return vocab


def _read(path: Path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        yield from csv.DictReader(line.replace("\x00", "") for line in f)


def _label_ids(field: str) -> list[int]:
    return [int(float(x)) for x in str(field or "").split(";") if x.strip()]


def load_tables(dataset: str, split: str = "test") -> tuple[list[dict], list[str]]:
    """Returns (tables, label_set). Each table: table_id, table_raw (row-major, <=6 rows), num_col,
    col_ids (column_index per column), gold_types ('-1' = not asked, else the gold names joined by ';' in gold order),
    gold_labels (the same as a list of names per column, [] = not asked).

    split="test" (also "train" / "valid"): the whole table with ALL its columns is the context (as in the original
    RACOON); every labeled column of that file is asked, whatever its number of gold types (a table is not known to be
    single- or multi-label beforehand); tables without a labeled column are left out.
    split="all": the three files together."""
    if dataset not in _DATASETS:
        raise ValueError(f"unsupported dataset {dataset!r}, expected one of {list(_DATASETS)}")
    if split != "all" and split not in _SPLITS:
        raise ValueError(f"unsupported split {split!r}, expected one of {list(_SPLITS) + ['all']}")
    base = ROOT / "data" / _DATASETS[dataset]
    vocab = _vocab(base)
    assert not any(";" in n for n in vocab.values()), "a label name contains ';', the gold separator"
    csv.field_size_limit(sys.maxsize)

    real_cells = None
    if dataset == "sotab":
        if split != "test":
            raise ValueError("sotab: the real cells exist for the test split only")
        import sotab_cells
        real_cells = sotab_cells.load("test")

    by_table: dict[str, dict[int, tuple[list[str], list[str]]]] = {}
    for name in (_SPLITS if split == "all" else (split,)):
        gold: dict[tuple[str, int], list[str]] = {}
        for setting in ("single_label", "multi_label"):  # union = every labeled column once, gold order kept
            path = base / setting / f"{name}.csv"
            if path.exists():
                for r in _read(path):
                    ids = [i for i in _label_ids(r["label"]) if i >= 0]
                    if ids:
                        gold[(r["table_id"], int(r["column_index"]))] = [vocab.get(i, str(i)) for i in ids]
        for r in _read(base / f"{name}.csv"):
            key = (r["table_id"], int(r["column_index"]))
            if real_cells is not None:
                cells = list(real_cells.get(r["table_id"], {}).get(key[1], []))[:MAX_ROWS]
            else:
                cells = (r["data"] or "").split(";")[:MAX_ROWS]
            by_table.setdefault(r["table_id"], {}).setdefault(key[1], (cells, gold.get(key, [])))

    tables = []
    for table_id, cols in by_table.items():
        col_ids = sorted(cols)
        labels = [cols[c][1] for c in col_ids]
        if not any(labels):
            continue  # nothing to ask in this table
        columns = [cols[c][0] for c in col_ids]
        n_rows = max(1, max((len(c) for c in columns), default=0))
        tables.append({
            "table_id": table_id,
            "table_raw": [[(columns[c][r] if r < len(columns[c]) else "") for c in range(len(columns))]
                          for r in range(n_rows)],
            "num_col": len(col_ids),
            "col_ids": col_ids,
            "gold_types": [";".join(l) if l else "-1" for l in labels],
            "gold_labels": labels,
        })
    return tables, [vocab[k] for k in sorted(vocab)]
