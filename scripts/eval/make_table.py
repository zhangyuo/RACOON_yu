# -*- coding: utf-8 -*-
"""
One row per (dataset, setting) in the layout of the results sheet:
    Dataset | Query number | Micro F1 | Macro F1 | Weighted F1 | Hits@1 | Token   (F1 and Hits@1 in %, Token = tokens per column)
read from the <result>.metrics.<setting>.json files written by eval_racoon.py.

Usage: make_table.py "<label>=<metrics.json>" ...
"""
import json
import sys

rows = []
for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    m = json.load(open(path))
    rows.append([label, m["total_samples"], f'{100 * m["micro_f1"]:.2f}', f'{100 * m["macro_f1"]:.2f}',
                 f'{100 * m["weighted_f1"]:.2f}', f'{100 * m["hits_at_1"]:.2f}', f'{m["cost"]["avg_tokens_per_col"]:.2f}'])
head = ["Dataset", "Query number", "Micro F1", "Macro F1", "Weighted F1", "Hits@1", "Token"]
print("| " + " | ".join(head) + " |")
print("|" + "---|" * len(head))
for r in rows:
    print("| " + " | ".join(str(x) for x in r) + " |")
print()
print("\t".join(head))
for r in rows:
    print("\t".join(str(x) for x in r))
