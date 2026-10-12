# -*- coding: utf-8 -*-
"""
The QID scan can run as several parallel jobs, each writing qid_list.<tag>.json (see build_qid_list.py).
load_all_qids() returns the union of qid_list.json and all qid_list.<tag>.json (progress and smoke-test files excluded).

Usage:  python merge_qid_lists.py     # writes the union to qid_list.json
"""
import json
import os
from pathlib import Path

SRC = Path(__file__).resolve().parent


def result_files():
    return [p for p in sorted(SRC.glob("qid_list*.json"))
            if "progress" not in p.name and "smoke" not in p.name]


def load_all_qids():
    merged = {}
    for p in result_files():
        with open(p, encoding="utf-8") as f:
            for qid, example in json.load(f).items():
                merged.setdefault(qid, example)
    return merged


if __name__ == "__main__":
    files = result_files()
    merged = load_all_qids()
    out = SRC / "qid_list.json"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)
    os.replace(tmp, out)
    print(f"merged {len(files)} files ({', '.join(p.name for p in files)}) -> {len(merged)} distinct QIDs in {out.name}")
