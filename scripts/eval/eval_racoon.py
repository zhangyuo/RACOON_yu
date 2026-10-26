# -*- coding: utf-8 -*-
"""
Evaluate a RACOON result file with the PROJECT's metric code: scripts/eval/rag4cta_metrics/ is a copy of src/eval/metrics.py and
src/label_setting.py of the RAG4CTA `test` branch (commit 26cbf0c), so every number comes from the same code as for the other methods.

    single_label : one predicted type per column; a hit on ANY gold type makes the gold the prediction, else the gold is the
                   first gold type (RACOON protocol)
    multi_label  : multi-hot prediction against the multi-hot gold

This script only translates the result file into the records compute_metrics expects and prints EVERY metric it returns.

Input: a result file with 10 fields per row (see RACOON_CTA.py):
    table_loop_id, pos, pred, hint, raw_answer, prompt_tokens, completion_tokens, table_id, col_id, gold
where `gold` is the gold type names joined by ';' in gold order and `pred` (multi_label) the predicted type names joined by ';'.
Use scripts/eval/merge_shards.py first when the run was sharded.

Usage: eval_racoon.py <result.csv> <single_label|multi_label>
Writes <result>.metrics.<setting>.json (all metrics incl. the per-type table).
"""
from __future__ import annotations  # keeps `list[dict]` hints safe on Python 3.8 too

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag4cta_metrics.metrics import compute_metrics  # noqa: E402

csv.field_size_limit(sys.maxsize)


def load_records(csv_path: str, setting: str) -> list[dict]:
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 10:
                raise ValueError(f"{csv_path}: a row has {len(row)} fields, expected 10 (table_id, col_id and gold are needed)")
            golds = [g for g in row[9].split(";") if g]
            rec = {
                "table_id": row[7],
                "col_id": row[8],
                "gold_types": golds,
                "gold_type": golds[0] if golds else "-1",
                "prompt_tokens": int(row[5]),
                "completion_tokens": int(row[6]),
                "label_setting": setting,
            }
            if setting == "multi_label":
                preds = [p for p in row[2].split(";") if p]
                rec["pred_types"] = preds
                rec["pred_type"] = preds[0] if preds else ""
            else:
                rec["pred_type"] = row[2]
            records.append(rec)
    return records


def main(csv_path: str, setting: str) -> None:
    assert setting in ("single_label", "multi_label"), setting
    records = load_records(csv_path, setting)
    metrics = compute_metrics(records, setting)
    print(f"{csv_path}\nsetting: {setting} | rows read: {len(records)}")
    for k, v in metrics.items():
        if k == "per_type":
            continue
        if isinstance(v, dict):
            print(f"  {k}: " + ", ".join(f"{kk}={vv}" for kk, vv in v.items()))
        else:
            print(f"  {k}: {v}")
    per_type = metrics.get("per_type", {})
    print(f"  per_type: {len(per_type)} types (precision / recall / f1 / support in the json file)")
    out_path = Path(csv_path).with_suffix(f".metrics.{setting}.json")
    json.dump(metrics, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"  all metrics written to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
