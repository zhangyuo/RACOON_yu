# -*- coding: utf-8 -*-
"""
Fill wikidata_cache.json["entity_full"] for every QID in qid_list.json (and the per-job qid_list.<tag>.json files), 50 QIDs per API request
(see wikidata_api.py). QIDs that are already cached with the full record are skipped, so it can be re-run
and resumed. utils.py reads the same file, so RACOON never has to query Wikidata live for these QIDs.

Usage:  python -u cache_wikidata_batch.py
"""
import json
import os
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from merge_qid_lists import load_all_qids  # noqa: E402
from wikidata_api import fetch_entities  # noqa: E402

QID_LIST = SRC / "qid_list.json"
CACHE = SRC / "wikidata_cache.json"
CHUNK = 200  # QIDs per saved step


def save(cache):
    tmp = CACHE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    os.replace(tmp, CACHE)  # never leave a half-written cache behind


def main():
    qids = list(load_all_qids())  # union of qid_list.json and the per-job qid_list.<tag>.json files
    cache = json.load(open(CACHE, encoding="utf-8")) if CACHE.exists() else {}
    cache.setdefault("entity_full", {})
    cache.setdefault("relation", {})
    full = cache["entity_full"]
    todo = [q for q in qids if q not in full or "type_hierarchy" not in full[q]]  # old SPARQL entries lack the new fields
    print(f"{len(qids)} QIDs in the list, {len(qids) - len(todo)} already cached, {len(todo)} to fetch", flush=True)

    t0, missing = time.time(), []
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        recs = fetch_entities(chunk)
        full.update(recs)
        missing += [q for q in chunk if q not in recs]
        save(cache)
        print(f"{min(i + CHUNK, len(todo))}/{len(todo)} fetched, {len(missing)} not found, {time.time() - t0:.0f}s", flush=True)
    if missing:
        print("QIDs that do not exist on Wikidata (not cached):", missing[:20], flush=True)
    print(f"done: {len(full)} entities in {CACHE}", flush=True)


if __name__ == "__main__":
    main()
