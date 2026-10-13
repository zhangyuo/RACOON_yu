# -*- coding: utf-8 -*-
"""
Verify wikidata_cache.json against the QID list(s) it was built for.

  1. completeness : every QID of the list is cached, or recorded in cache["missing"] (deleted on Wikidata). Nothing else.
  2. structure    : every record has the fields utils.py reads, and they agree with each other.
  3. cross-check  : a random sample is fetched again through the Wikidata MCP service (a different implementation than the
                    Action API used to build the cache) and compared: label, description and the set of P31 types.

Usage:  python check_wikidata_cache.py [--tags baseline] [--sample 100] [--seed 0] [--no_mcp]
Exit code 1 if a completeness or structure error, or a type mismatch in the cross-check, was found.
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from merge_qid_lists import load_all_qids  # noqa: E402

CACHE = SRC / "wikidata_cache.json"
REQUIRED = {"label": str, "description": str, "qids": list, "entity_set": list, "raw_bindings": list,
            "aliases": list, "subclass_of": list, "wikipedia_title": str, "statements": list, "type_hierarchy": dict}
MCP = "https://wd-mcp.wmcloud.org/tool/get_statements?entity_id={qid}&lang=en"


def check_structure(qid, r):
    errs = []
    for k, t in REQUIRED.items():
        if k not in r:
            errs.append(f"{qid}: missing field {k}")
        elif not isinstance(r[k], t):
            errs.append(f"{qid}: field {k} is {type(r[k]).__name__}, expected {t.__name__}")
    if errs:
        return errs
    if len(r["qids"]) != len(r["raw_bindings"]):
        errs.append(f"{qid}: len(qids)={len(r['qids'])} != len(raw_bindings)={len(r['raw_bindings'])}")
    labels = set()
    for b, q in zip(r["raw_bindings"], r["qids"]):
        if not re.fullmatch(r"Q\d+", b.get("type_qid", "")) or not b.get("type_label"):
            errs.append(f"{qid}: bad binding {b}")
            continue
        used = (b["pq_obj_qid"], b["pq_obj_label"]) if "pq_obj_qid" in b else (b["type_qid"], b["type_label"])
        if q != used[0]:
            errs.append(f"{qid}: qids entry {q} does not match its binding {b}")
        labels.add(used[1])
    if set(r["entity_set"]) != labels:
        errs.append(f"{qid}: entity_set {sorted(r['entity_set'])[:3]} != labels of the bindings {sorted(labels)[:3]}")
    for tq in {b.get("type_qid") for b in r["raw_bindings"]}:
        if tq not in r["type_hierarchy"]:
            errs.append(f"{qid}: type {tq} has no type_hierarchy entry")
    return errs


def mcp_statements(qid, retries=4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(MCP.format(qid=qid), method="POST")
            with urllib.request.urlopen(req, timeout=40) as resp:
                return json.loads(resp.read())["result"]
        except urllib.error.HTTPError as e:
            if e.code != 429:
                return None
            time.sleep(3 * (attempt + 1))
        except Exception:
            time.sleep(3)
    return None


def parse_mcp(text, qid):
    """-> (label, description, set of P31 type QIDs) from the get_statements text."""
    label = desc = None
    types = set()
    for line in text.splitlines():
        m = re.match(rf"(.*) \({qid}\): (.*)$", line)
        if not m:
            continue
        label, rest = m.group(1), m.group(2)
        if rest.startswith("description: "):
            desc = rest[len("description: "):]
        elif rest.startswith("instance of (P31): "):
            t = re.search(r"\((Q\d+)\)$", rest)
            if t:
                types.add(t.group(1))
    return label, desc, types


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=None)
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_mcp", action="store_true")
    args = ap.parse_args()

    qids = list(load_all_qids(args.tags.split(",") if args.tags else None))
    cache = json.load(open(CACHE, encoding="utf-8"))
    full, gone = cache.get("entity_full", {}), set(cache.get("missing", []))
    problems = 0

    print(f"== 1. completeness: {len(qids)} QIDs in the list")
    cached = [q for q in qids if q in full]
    deleted = [q for q in qids if q in gone and q not in full]
    unaccounted = [q for q in qids if q not in full and q not in gone]
    print(f"   cached: {len(cached)} | not on Wikidata (deleted): {len(deleted)} | unaccounted: {len(unaccounted)}")
    if unaccounted:
        problems += 1
        print("   FAIL, not cached and not marked missing, e.g.", unaccounted[:10])

    print(f"== 2. structure of {len(cached)} records")
    errs = [e for q in cached for e in check_structure(q, full[q])]
    if errs:
        problems += 1
        print(f"   FAIL, {len(errs)} problems, first ones:")
        for e in errs[:10]:
            print("    ", e)
    else:
        print("   ok")
    st = Counter()
    for q in cached:
        r = full[q]
        st["label is just the QID (no English label)"] += r["label"] == q
        st["no description"] += not r["description"]
        st["no P31 type (qids empty)"] += not r["qids"]
        st["with a P642 qualifier"] += any("pq_obj_qid" in b for b in r["raw_bindings"])
        st["with a Wikipedia page"] += bool(r["wikipedia_title"])
    for k, v in st.items():
        print(f"   {k}: {v} ({100 * v / max(len(cached), 1):.1f}%)")

    if args.no_mcp or not cached:
        print("== 3. cross-check skipped")
    else:
        sample = random.Random(args.seed).sample(cached, min(args.sample, len(cached)))
        print(f"== 3. cross-check of {len(sample)} random QIDs against the Wikidata MCP service")
        compared, bad_types, bad_label, bad_desc, unavailable, extra_ranks = 0, [], [], [], 0, 0
        for i, q in enumerate(sample):
            if i % 10 == 0:
                print(f"   ... {i}/{len(sample)} checked", flush=True)
            text = mcp_statements(q)
            time.sleep(1.5)  # the service answers 429 to bursts
            if not text:
                unavailable += 1
                continue
            label, desc, types = parse_mcp(text, q)
            if label is None:
                unavailable += 1
                continue
            compared += 1
            r = full[q]
            ours = {b["type_qid"] for b in r["raw_bindings"]}
            if types - ours:  # a type the MCP service lists is missing from the cache: a real error
                bad_types.append((q, sorted(types), sorted(ours)))
            elif ours - types:  # MCP lists only the best-ranked P31 values, the cache keeps all ranks (as the SPARQL query did)
                extra_ranks += 1
            if label != r["label"]:
                bad_label.append((q, label, r["label"]))
            if (desc or "") != r["description"]:
                bad_desc.append((q, desc, r["description"]))
        print(f"   compared {compared} (MCP gave no answer for {unavailable})")
        print(f"   P31 types missing from the cache: {len(bad_types)} | label differs: {len(bad_label)} | description differs: {len(bad_desc)}")
        print(f"   cache has extra lower-rank P31 types (expected, MCP lists only the best rank): {extra_ranks}")
        for name, rows in (("types", bad_types), ("label", bad_label), ("description", bad_desc)):
            for row in rows[:3]:
                print(f"     {name}: {row}")
        if bad_types:
            problems += 1

    print("RESULT:", "PROBLEMS FOUND" if problems else "all checks passed")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
