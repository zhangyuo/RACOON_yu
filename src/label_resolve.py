# -*- coding: utf-8 -*-
"""
Map what the model wrote back to a label of the label set (RAG4CTA addition, shared by RACOON_CTA.py and postprocess_resolve.py).

A prediction that is not in the label set is replaced only when exactly ONE label matches it, in one of these ways (both sides
compared lower case, underscores as spaces):
  1. the label without its trailing wikidata id: "genus" -> "genus (Q34740)"
  2. the last part of a "domain.type" label:      "film"  -> "film.film",  "organization" -> "organization.organization"
Anything ambiguous or without a match stays as the model wrote it.
"""
from __future__ import annotations  # RACOON runs on Python 3.8

import re


def _norm(x: str) -> str:
    return " ".join(str(x).lower().replace("_", " ").split())


def _no_qid(x: str) -> str:
    return _norm(re.sub(r"\s*\(Q\d+\)\s*$", "", str(x)))


def _last_part(x: str) -> str:
    return _norm(str(x).rsplit(".", 1)[-1])


_index_cache: dict = {}


def _indexes(label_set):
    key = id(label_set)
    if key not in _index_cache:
        by_qid, by_last = {}, {}
        for l in label_set:
            by_qid.setdefault(_no_qid(l), []).append(l)
            if "." in str(l):
                by_last.setdefault(_last_part(l), []).append(l)
        _index_cache[key] = (by_qid, by_last, set(label_set))
    return _index_cache[key]


def resolve_to_label_set(pred, label_set):
    """pred: a string or a list of strings."""
    if isinstance(pred, list):
        return [resolve_to_label_set(p, label_set) for p in pred]
    by_qid, by_last, members = _indexes(label_set)
    if pred in members:
        return pred
    for index, key in ((by_qid, _no_qid(pred)), (by_last, _norm(pred))):
        hit = index.get(key, [])
        if len(hit) == 1:
            return hit[0]
    return pred
