# -*- coding: utf-8 -*-
"""
Evaluate a RACOON+ (sc/db) output CSV with RAG4CTA's own metric definitions,
instead of RACOON's own TURL-only CTA_eval.py -- so the number is directly
comparable to every other method in docs/RESULTS.md.

The output CSV rows are (see RACOON_CTA.py's writer for dataset != 'turl'):
    table_loop_id, pos, pred_type, hint, raw_prediction,
    prompt_tokens, completion_tokens, table_id, col_id, gold_type

Usage: eval_racoon.py <output.csv>
"""
from __future__ import annotations  # keeps `list[dict]` hints safe on Python 3.8 too

import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getenv("RAG4CTA_ROOT", os.path.expanduser("~/RAG4CTA")))  # holds src/eval/metrics.py
from src.eval.metrics import compute_metrics  # noqa: E402


def load_records(csv_path: str) -> list[dict]:
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 10:
                continue  # not a sc/db row (e.g. a turl-format row, fewer columns)
            records.append({
                "pred_type": row[2],
                "prompt_tokens": int(row[5]),
                "completion_tokens": int(row[6]),
                "table_id": row[7],
                "col_id": row[8],
                "gold_type": row[9],
            })
    return records


def main(csv_path: str) -> None:
    records = load_records(csv_path)
    metrics = compute_metrics(records)
    print(f"rows read: {len(records)}  scored: {metrics.get('total_samples')}")
    print(f"Micro F1: {metrics['micro_f1']}")
    print(f"Macro F1: {metrics['macro_f1']}")
    print(f"avg tokens/col: prompt={metrics['cost']['avg_prompt_tokens_per_col']}"
          f"  completion={metrics['cost']['avg_completion_tokens_per_col']}"
          f"  total={metrics['cost']['avg_tokens_per_col']}")
    out_path = Path(csv_path).with_suffix(".metrics.json")
    json.dump(metrics, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"full metrics written to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
