# -*- coding: utf-8 -*-
"""
CTA evaluation metrics, in two settings (following RACOON):
  - single_label: one predicted type; a hit on ANY gold label counts as correct
  - multi_label : multi-hot prediction vs. multi-hot gold over the type vocabulary

Metrics:
  - Micro / Macro / Weighted F1 (+ samples F1 for multi_label)
  - Per-type precision / recall / F1
  - Confusion summary

All computed from JSONL log files.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, precision_score, recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

from .label_setting import LABEL_SETTINGS, MULTI_LABEL, SINGLE_LABEL


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _gold_types(r: dict) -> list[str]:
    """All gold labels of a record (legacy logs only carry gold_type)."""
    golds = r.get("gold_types") or [r.get("gold_type")]
    return [str(g) for g in golds if g is not None and str(g) != "-1"]


def _pred_types(r: dict) -> list[str]:
    preds = r.get("pred_types") or [r.get("pred_type")]
    return [str(p) for p in preds if p]


def _labeled_columns(records: list[dict]) -> list[dict]:
    """
    Keep labeled records, one per column.  Legacy logs expanded multi-label
    columns into one record per gold label; those rows are merged back into a
    single column (first record's prediction, all gold labels in order).
    """
    columns: dict = {}
    for i, r in enumerate(records):
        golds = _gold_types(r)
        if not golds:
            continue
        key = (r.get("table_id"), r.get("col_id")) if "gold_types" not in r else ("_row", i)
        if key in columns:
            merged = columns[key]["gold_types"]
            merged.extend(g for g in golds if g not in merged)
        else:
            columns[key] = {**r, "gold_types": list(dict.fromkeys(golds))}
    return list(columns.values())


def _topk_hit(r: dict, golds: list[str]) -> bool:
    """Top-k hit using `top_k` (list[dict|str]) or `predicted_candidates`."""
    labels: list[str] = []
    for item in r.get("top_k", []):
        if isinstance(item, dict):
            if "label" in item:
                labels.append(str(item["label"]))
        elif item is not None:
            labels.append(str(item))
    if not labels:
        labels = [str(x) for x in r.get("predicted_candidates", []) if x is not None]
    return any(g in labels for g in golds)


def _ranked_preds(r: dict) -> list[str]:
    """
    Ranked predictions of a record: pred_type, then the rest of pred_types
    (best first), then any remaining top-k candidates.
    """
    ranked: list[str] = []
    candidates = [r.get("pred_type")] + list(r.get("pred_types") or [])
    for item in r.get("top_k") or []:
        candidates.append(item.get("label") if isinstance(item, dict) else item)
    for c in candidates:
        if c and str(c) not in ranked:
            ranked.append(str(c))
    return ranked


def hits_at_1(labeled: list[dict]) -> float:
    """Fraction of columns whose top-1 prediction is one of the gold types.
    Defined the same way in both settings, so they can be compared."""
    return sum(1 for r in labeled if (_ranked_preds(r) or [None])[0] in r["gold_types"]) / len(labeled)


def recall_at_gt(labeled: list[dict]) -> float:
    """Mean over columns of |top-|G| ranked predictions ∩ G| / |G|."""
    total = 0.0
    for r in labeled:
        golds = set(r["gold_types"])
        top = _ranked_preds(r)[:len(golds)]
        total += len(golds.intersection(top)) / len(golds)
    return total / len(labeled)


def _per_type(report: dict) -> dict[str, dict]:
    per_type: dict[str, dict] = {}
    for label, stats in report.items():
        if label in {"accuracy", "micro avg", "macro avg", "weighted avg", "samples avg"}:
            continue
        per_type[label] = {
            "precision": round(float(stats.get("precision", 0.0)), 4),
            "recall": round(float(stats.get("recall", 0.0)), 4),
            "f1": round(float(stats.get("f1-score", 0.0)), 4),
            "support": int(stats.get("support", 0)),
        }
    return per_type


def _single_label_scores(labeled: list[dict]) -> dict:
    """
    RACOON single-label protocol: one predicted type per column.  If it is any
    of the gold types, the gold is replaced by the prediction (perfect match);
    otherwise the gold falls back to the FIRST gold type.
    hits_at_1 = prediction ∈ gold types (equals accuracy under this protocol).
    """
    y_true: list[str] = []
    y_pred: list[str] = []
    for r in labeled:
        golds = r["gold_types"]
        pred = str(r.get("pred_type") or "")
        y_pred.append(pred)
        y_true.append(pred if pred in golds else golds[0])

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "micro_f1": round(f1_score(y_true, y_pred, average="micro", zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "hits_at_1": round(hits_at_1(labeled), 4),
        "correct": sum(1 for t, p in zip(y_true, y_pred) if t == p),
        "per_type": _per_type(report),
    }


def _multi_label_scores(labeled: list[dict]) -> dict:
    """
    RACOON multi-label protocol: gold and prediction are multi-hot vectors
    over the types; every type counts as TP / FP / FN / TN.

      micro_*    pooled TP / FP / FN over all columns and types (RACOON's
                 reported metric); FPs on any type are penalised here
      macro_f1   mean per-type F1 over the types present in the gold labels
                 (types only ever predicted would otherwise each add F1 = 0)
      samples_f1 mean per-column F1
      accuracy   exact-match (subset) accuracy
      recall_at_gt  mean over columns of |top-|G| ranked preds ∩ G| / |G|
      hits_at_1     top-1 prediction ∈ G (same definition as single-label,
                    so the two settings are directly comparable)
    """
    gold_sets = [r["gold_types"] for r in labeled]
    pred_sets = [_pred_types(r) for r in labeled]
    classes = sorted({t for s in gold_sets + pred_sets for t in s})
    mlb = MultiLabelBinarizer(classes=classes)
    y_true = mlb.fit_transform(gold_sets)
    y_pred = mlb.transform(pred_sets)
    gold_type_idx = [i for i in range(len(classes)) if y_true[:, i].any()]

    report = classification_report(y_true, y_pred, target_names=classes,
                                   output_dict=True, zero_division=0)
    return {
        "micro_f1": round(f1_score(y_true, y_pred, average="micro", zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=gold_type_idx,
                                   average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, labels=gold_type_idx,
                                      average="weighted", zero_division=0), 4),
        "samples_f1": round(f1_score(y_true, y_pred, average="samples", zero_division=0), 4),
        "micro_precision": round(precision_score(y_true, y_pred, average="micro", zero_division=0), 4),
        "micro_recall": round(recall_score(y_true, y_pred, average="micro", zero_division=0), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "recall_at_gt": round(recall_at_gt(labeled), 4),
        "hits_at_1": round(hits_at_1(labeled), 4),
        "correct": sum(1 for g, p in zip(gold_sets, pred_sets) if set(g) == set(p)),
        "avg_gold_labels": round(sum(map(len, gold_sets)) / len(gold_sets), 4),
        "avg_pred_labels": round(sum(map(len, pred_sets)) / len(pred_sets), 4),
        "n_gold_types": len(gold_type_idx),
        "n_pred_only_types": len(classes) - len(gold_type_idx),
        "per_type": _per_type(report),
    }


def compute_metrics(records: list[dict], label_setting: str | None = None) -> dict:
    """
    Compute CTA metrics from a list of log records.

    Each record must have 'gold_type' (or 'gold_types') and 'pred_type'
    (plus 'pred_types' in the multi-label setting).
    Records without gold labels, or with gold_type == -1 (unlabeled columns),
    are excluded — following the REVEAL benchmark convention (class_id > -1).

    label_setting: "single_label" | "multi_label" (see src/label_setting.py).
    Defaults to the setting recorded in the log, else single_label.
    """
    labeled = _labeled_columns(records)
    if not labeled:
        return {"error": "no labeled records found"}

    setting = label_setting or labeled[0].get("label_setting") or SINGLE_LABEL
    if setting not in LABEL_SETTINGS:
        raise ValueError(f"Unknown label_setting: {setting}")
    scores = _multi_label_scores(labeled) if setting == MULTI_LABEL else _single_label_scores(labeled)

    topk_hits = [_topk_hit(r, r["gold_types"]) for r in labeled]
    total = len(labeled)
    n_multi = sum(1 for r in labeled if len(r["gold_types"]) > 1)

    # Cost & latency summary
    total_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in records)
    total_completion_tokens = sum(r.get("completion_tokens", 0) for r in records)
    total_tokens = total_prompt_tokens + total_completion_tokens
    n_queries = len(records) if records else 1
    avg_latency = (
        sum(r.get("latency_ms", 0) for r in records) / n_queries
        if records else 0
    )
    avg_tokens_per_query = total_tokens / n_queries if n_queries > 0 else 0

    # Retrieval stats
    retrieve_policies = Counter(r.get("retrieve_policy", "") for r in records)
    conflict_rate = sum(1 for r in records if r.get("conflict_detected")) / len(records) if records else 0
    # Share of columns whose pred_type is not the model's highest-confidence candidate
    topk_rank_conflict_rate = sum(1 for r in labeled if r.get("topk_rank_conflict")) / total

    per_type = scores.pop("per_type")
    return {
        "label_setting": setting,
        **scores,
        "top_k_accuracy": round(sum(topk_hits) / total, 4),
        "total_samples": total,
        "multi_label_columns": n_multi,
        "total_all": len(records),
        "unlabeled_skipped": sum(1 for r in records if not _gold_types(r)),
        "per_type": per_type,
        "cost": {
            "total_tokens": total_tokens,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "avg_tokens_per_col": round(avg_tokens_per_query, 2),
            "avg_tokens_per_query": round(avg_tokens_per_query, 2),
            "avg_prompt_tokens_per_col": round(total_prompt_tokens / n_queries, 2),
            "avg_prompt_tokens_per_query": round(total_prompt_tokens / n_queries, 2),
            "avg_completion_tokens_per_col": round(total_completion_tokens / n_queries, 2),
            "avg_completion_tokens_per_query": round(total_completion_tokens / n_queries, 2),
        },
        "latency": {"avg_ms": round(avg_latency, 2)},
        "retrieval_policy_distribution": dict(retrieve_policies),
        "conflict_rate": round(conflict_rate, 4),
        "topk_rank_conflict_rate": round(topk_rank_conflict_rate, 4),
    }


def evaluate_file(log_path: str | Path, label_setting: str | None = None) -> dict:
    records = load_jsonl(log_path)
    return compute_metrics(records, label_setting)
