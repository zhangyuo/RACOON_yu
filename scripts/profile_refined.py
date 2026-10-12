# -*- coding: utf-8 -*-
"""
Where does the ReFinED scan spend its time? Loads ReFinED the way build_qid_list.py does, links a few tiny tables
and prints the device in use, the time per call and a cProfile summary.
Run from the RACOON repo root on a GPU node:  python scripts/profile_refined.py
"""
import cProfile
import os
import pstats
import sys
import time

os.environ.setdefault("VLLM_API_KEY", "unused")
sys.path.insert(0, "src")

t0 = time.time()
from KG_Linker import get_column_wise_spans, refined  # noqa: E402

print(f"load: {time.time() - t0:.0f}s", flush=True)
try:
    import torch
    print("cuda available:", torch.cuda.is_available(), "| model device:", next(refined.model.parameters()).device,
          "| refined.device:", getattr(refined, "device", None), flush=True)
except Exception as e:  # attribute names differ between ReFinED versions
    print("device check failed:", e, flush=True)

rows = [["Paris", "France", "Europe"], ["London", "United Kingdom", "Europe"], ["Berlin", "Germany", "Europe"]] * 3
cols = [list(c) for c in zip(*rows)]
for i in range(4):
    t = time.time()
    get_column_wise_spans("col", rows, cols)
    print(f"call {i}: {time.time() - t:.2f}s", flush=True)

pr = cProfile.Profile()
pr.enable()
for _ in range(5):
    get_column_wise_spans("col", rows, cols)
pr.disable()
pstats.Stats(pr).sort_stats("cumulative").print_stats(20)
