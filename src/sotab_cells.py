# -*- coding: utf-8 -*-
"""
Real cell boundaries of SOTAB-CTA for RACOON.

The shared file data/SOTAB-CTA/test.csv joins the cells of a column with a plain space, so "first 6 cells" cannot be recovered
from it. The tables themselves are public (SOTAB V1, CTA_Test.zip, https://data.dws.informatik.uni-mannheim.de/structureddata/sotab/CTA_Test.zip; one gzip
JSON-lines file per table, one JSON object per row, keys = column positions). The shared test split is the V1 CTA test set
(V2 has only 609 test tables); the shared train and valid files are not rebuilt here, RACOON only needs test. The table_id of the shared files is the official file name, so the cells
are read from there:

    python sotab_cells.py [split=test]      ->  <OFFICIAL_DIR>/<split>_first6_cells.json
    {table_id: {column_index: [cell, ...]}}  (first MAX_ROWS non-empty cells of EVERY column of the tables of the split)

A cell that is a list is written as its items joined by a space; null and empty cells are dropped. On the tables of the shared test.csv this reproduces the shared "data" text of 97.5% of the columns exactly
(the rest is cut off in the shared file).

Env var RAG4CTA_ROOT: checkout that holds data/ (default ~/RAG4CTA); OFFICIAL_DIR: folder with the official zip.
"""
from __future__ import annotations  # RACOON runs on Python 3.8

import csv
import gzip
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(os.getenv("RAG4CTA_ROOT", os.path.expanduser("~/RAG4CTA")))
OFFICIAL_DIR = Path(os.getenv("OFFICIAL_DIR", "/nfs/srv/data2/af62lp@cs.aau.dk/cta_data/raw/SOTAB_official_v1"))
ZIPS = ("CTA_Test.zip",)  # SOTAB V1 CTA test set: 7,026 tables, 15,040 annotated columns = the shared test split
MAX_ROWS = 6


def flat(v) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return " ".join(str(x) for x in v)
    return str(v)


def first_cells(lines, max_rows: int = MAX_ROWS) -> dict[int, list[str]]:
    cols: dict[int, list[str]] = {}
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        for k, v in row.items():
            cell = flat(v)
            c = cols.setdefault(int(k), [])
            if cell and len(c) < max_rows:
                c.append(cell)
    return cols


def build(split: str = "test") -> Path:
    csv.field_size_limit(sys.maxsize)
    need: set[str] = set()
    with open(ROOT / "data" / "SOTAB-CTA" / f"{split}.csv", newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(line.replace("\x00", "") for line in f):
            need.add(r["table_id"])
    out: dict[str, dict[int, list[str]]] = {}
    for name in ZIPS:
        with zipfile.ZipFile(OFFICIAL_DIR / name) as z:
            for member in z.namelist():
                tid = member.split("/")[-1]
                if member.endswith(".gz") and tid in need and tid not in out:
                    raw = gzip.decompress(z.read(member)).decode("utf-8", errors="replace")
                    out[tid] = first_cells(raw.splitlines())
        print(f"{name}: {len(out)}/{len(need)} tables found so far", flush=True)
    missing = sorted(need - set(out))
    print(f"tables of the {split} split: {len(need)}, found in the official files: {len(out)}, missing: {len(missing)} {missing[:3]}")
    path = OFFICIAL_DIR / f"{split}_first6_cells.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    return path


def load(split: str = "test") -> dict[str, dict[int, list[str]]]:
    with open(OFFICIAL_DIR / f"{split}_first6_cells.json", encoding="utf-8") as f:
        return {t: {int(c): v for c, v in cols.items()} for t, cols in json.load(f).items()}


if __name__ == "__main__":
    print("written", build(sys.argv[1] if len(sys.argv) > 1 else "test"))
