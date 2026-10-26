# -*- coding: utf-8 -*-
"""Evaluate a RACOON result file only on the columns of one data folder (collaborator's protocol): the single_label setting on the
columns of <dataset>/single_label/test.csv (exactly 1 gold), the multi_label setting on those of <dataset>/multi_label/test.csv (>=2 golds).
Usage: eval_by_folder.py <dataset_dir_name> <result.csv> <single_label|multi_label>   -> <result>.byfolder.csv + eval_racoon.py metrics."""
import csv, os, subprocess, sys
csv.field_size_limit(sys.maxsize)
ds, res, setting = sys.argv[1:4]
path = os.path.expanduser(f"~/RAG4CTA/data/{ds}/{setting}/test.csv")
keys = set()
if os.path.exists(path):
    for r in csv.DictReader(open(path, newline="", encoding="utf-8", errors="replace")):
        if any(int(float(x)) >= 0 for x in r["label"].split(";") if x.strip()):
            keys.add((r["table_id"], str(int(r["column_index"]))))
rows = [r for r in csv.reader(open(res, newline="", encoding="utf-8")) if len(r) == 10]
got = {(r[7], r[8]) for r in rows}
sub = [r for r in rows if (r[7], r[8]) in keys]
print(f"{ds} {setting}: folder columns {len(keys)} | in result {len(keys & got)} | missing {len(keys - got)} | rows kept {len(sub)} (distinct {len({(r[7], r[8]) for r in sub})})")
out = res.replace(".csv", ".byfolder.csv")
csv.writer(open(out, "w", newline="", encoding="utf-8")).writerows(sub)
subprocess.run([os.environ.get("EVAL_PY", sys.executable), os.path.join(os.path.dirname(__file__), "eval_racoon.py"), out, setting], check=True)
