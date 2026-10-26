# -*- coding: utf-8 -*-
"""
Apply label_resolve.resolve_to_label_set to RACOON result files that were produced before the rule existed, WITHOUT asking the LLM
again: the prediction is re-derived from the stored raw answer (field 5), so only the mapping of a written type to a label changes.

    python postprocess_resolve.py <outputs/<model>/turl_freebase> <vocab type_vocab.txt>

For every <setting>/RACOON_col_type.shard*.csv: the original is moved to <setting>/before_resolve/, the resolved file takes its name.
Note: a column whose FIRST answer was invalid got a retry from the model; only the stored (last) answer is re-resolved here.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import shutil
import sys

from label_resolve import resolve_to_label_set

csv.field_size_limit(sys.maxsize)


def raw_types(raw):
    m = re.findall(r"FINAL ANSWER:\s*(\{.*\})", raw, flags=re.S) or re.findall(r"(\{.*\})", raw, flags=re.S)
    try:
        t = json.loads(m[-1].replace("'", '"'))["type"]
        return [t] if isinstance(t, str) else [str(x).strip() for x in t if str(x).strip()]
    except Exception:
        return None


def main(out_dir, vocab_path):
    label_set = [l.rstrip("\n").split("\t")[1].strip() for l in open(vocab_path, encoding="utf-8") if "\t" in l]
    members = set(label_set)
    for setting in ("single_label", "multi_label"):
        files = sorted(glob.glob(os.path.join(out_dir, setting, "RACOON_col_type.shard*.csv")))
        backup = os.path.join(out_dir, setting, "before_resolve")
        os.makedirs(backup, exist_ok=True)
        n = changed = empty_before = empty_after = oov_before = oov_after = 0
        for f in files:
            rows = list(csv.reader(open(f, newline="", encoding="utf-8")))
            out = []
            for r in rows:
                old = r[2]
                types = raw_types(r[4])
                if types:
                    res = resolve_to_label_set(types, label_set)
                    if setting == "multi_label":
                        new = ";".join(dict.fromkeys(t for t in res if t in members))
                    else:
                        new = res[0]
                else:
                    new = old
                n += 1
                changed += new != old
                empty_before += not old or old in ("None", "NA")
                empty_after += not new or new in ("None", "NA")
                oov_before += any(x not in members for x in old.split(";") if x) if old not in ("None", "NA") else 1
                oov_after += any(x not in members for x in new.split(";") if x) if new not in ("None", "NA") else 1
                out.append(r[:2] + [new] + r[3:])
            shutil.move(f, os.path.join(backup, os.path.basename(f)))
            with open(f, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows(out)
        print(f"{setting}: {len(files)} files, {n} columns, predictions changed {changed}, "
              f"empty/None {empty_before} -> {empty_after}, with a type outside the label set {oov_before} -> {oov_after}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
