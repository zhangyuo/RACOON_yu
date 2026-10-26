# -*- coding: utf-8 -*-
"""
Concatenate the shard files of one run into one result file (the shards are disjoint sets of tables).

Usage: merge_shards.py <result dir, e.g. outputs/<model>/sc/single_label>
Writes <dir>/RACOON_col_type.merged.csv from <dir>/RACOON_col_type.shard*of*.csv and checks that no column appears twice.
"""
import csv
import glob
import os
import sys

csv.field_size_limit(sys.maxsize)
d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "RACOON_col_type.shard*of*.csv")))
rows, seen = [], set()
for f in files:
    for r in csv.reader(open(f, newline="", encoding="utf-8")):
        key = (r[7], r[8]) if len(r) >= 10 else (f, r[0], r[1])
        assert key not in seen, f"column {key} appears twice"
        seen.add(key)
        rows.append(r)
out = os.path.join(d, "RACOON_col_type.merged.csv")
with open(out, "w", newline="", encoding="utf-8") as fh:
    csv.writer(fh).writerows(rows)
print(f"{len(files)} shard files -> {len(rows)} columns -> {out}")
