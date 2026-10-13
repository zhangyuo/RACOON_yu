# -*- coding: utf-8 -*-
"""
Evaluate a RACOON `--dataset turl` output CSV as multi-label Micro/Macro F1,
independent of RACOON's own CTA_eval.py -- used as a cross-check, and to
report TURL alongside SC/DB in a consistent format.

TURL's own ground truth is multi-label (a column can have several correct
types), over the paper's own 127-label reduced space -- NOT the 255-label
TURL-Wiki-CTA space RAG4CTA uses elsewhere, so src/eval/metrics.py's
single-label compute_metrics doesn't apply here.

Provenance: rather than reimplementing multi-label F1 from scratch, this
reuses REVEAL's own f1_score_multilabel (cta_baselines/PLM/REVEAL/src/utils.py)
-- the same, already-established function REVEAL's own TURL evaluation
(repro/scripts/gen_results_turl.py) calls on genuine, unmodified multi-hot
ground truth. Fed the same way here, so the RACOON-on-TURL number in this
project's results table is computed by the identical method as the REVEAL
numbers already in it, not a new ad hoc formula.

Ground truth alignment: CTA_remapped_test_labels.txt has one line per column,
in the exact same order RACOON_CTA.py's turl writer emits rows (table 0's
columns, then table 1's, ...) -- this is RACOON's own eval.py's assumption
too, and is checked here (raises if row counts don't match 1:1).

Usage: eval_racoon_turl.py <output.csv>
"""
from __future__ import annotations

import ast
import csv
import json
import os
import sys
from pathlib import Path

_RACOON_SRC = Path(__file__).resolve().parents[2] / "src"  # this repository
sys.path.insert(0, str(_RACOON_SRC / "data"))
from TURL_CTA_label_reduction import reduced_label_set  # noqa: E402

_REVEAL_SRC = Path(os.getenv("REVEAL_SRC", os.path.expanduser("~/cta_baselines/PLM/REVEAL/src")))
sys.path.insert(0, str(_REVEAL_SRC))
from utils import f1_score_multilabel  # noqa: E402  -- REVEAL's own metric, reused as-is


def load_gold() -> list[list[str]]:
    lines = open(_RACOON_SRC / "data" / "CTA_remapped_test_labels.txt").readlines()
    return [ast.literal_eval(line.strip()) for line in lines]


def load_preds(csv_path: str) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [{"pred_type": row[2], "prompt_tokens": int(row[5]), "completion_tokens": int(row[6])}
                for row in csv.reader(f) if len(row) >= 7]


def main(csv_path: str) -> None:
    gold = load_gold()
    preds = load_preds(csv_path)
    if len(gold) != len(preds):
        raise ValueError(f"row count mismatch: {len(gold)} gold rows vs {len(preds)} prediction rows "
                          "-- alignment assumption (one row per column, same order) is broken")

    labels = reduced_label_set
    idx = {t: i for i, t in enumerate(labels)}
    n = len(labels)

    y_true = [[0] * n for _ in gold]
    for row, types in zip(y_true, gold):
        for t in types:
            if t in idx:
                row[idx[t]] = 1

    y_pred = [[0] * n for _ in preds]
    n_illegal = 0
    for row, p in zip(y_pred, preds):
        if p["pred_type"] in idx:
            row[idx[p["pred_type"]]] = 1
        else:
            n_illegal += 1  # left as an all-zero prediction row, same as RACOON's own serialize()

    micro_f1, macro_f1, _class_f1, _conf_mat = f1_score_multilabel(y_true, y_pred)
    total_prompt = sum(p["prompt_tokens"] for p in preds)
    total_completion = sum(p["completion_tokens"] for p in preds)

    print(f"rows: {len(preds)}  illegal predictions: {n_illegal} ({n_illegal / len(preds):.1%})")
    print(f"Micro F1: {round(micro_f1, 4)}")
    print(f"Macro F1: {round(macro_f1, 4)}")
    print(f"avg tokens/col: prompt={total_prompt / len(preds):.2f}  "
          f"completion={total_completion / len(preds):.2f}  "
          f"total={(total_prompt + total_completion) / len(preds):.2f}")

    out = {
        "micro_f1": round(float(micro_f1), 4),
        "macro_f1": round(float(macro_f1), 4),
        "n_rows": len(preds),
        "n_illegal": n_illegal,
        "avg_prompt_tokens_per_col": round(total_prompt / len(preds), 2),
        "avg_completion_tokens_per_col": round(total_completion / len(preds), 2),
        "avg_tokens_per_col": round((total_prompt + total_completion) / len(preds), 2),
    }
    out_path = Path(csv_path).with_suffix(".metrics.json")
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"full metrics written to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
