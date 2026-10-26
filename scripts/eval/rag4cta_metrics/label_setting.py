# -*- coding: utf-8 -*-
"""
Single-label vs. multi-label CTA settings (following RACOON, Wei et al.).

  single_label: the model outputs exactly ONE type per column.  At evaluation
                time, if the prediction is any of the column's gold types the
                gold is replaced by the prediction (counted as a perfect match);
                otherwise the gold falls back to the FIRST gold type.
  multi_label:  the model outputs a SET of types per column.  Gold and
                prediction are multi-hot vectors over the type vocabulary and
                every candidate counts as TP / FP / FN / TN.

The setting is read from cfg["experiment"]["label_setting"].
"""

from __future__ import annotations

SINGLE_LABEL = "single_label"
MULTI_LABEL = "multi_label"
LABEL_SETTINGS = (SINGLE_LABEL, MULTI_LABEL)


def get_label_setting(cfg: dict | None) -> str:
    setting = str((cfg or {}).get("experiment", {}).get("label_setting", SINGLE_LABEL)).lower()
    if setting not in LABEL_SETTINGS:
        raise ValueError(f"Unknown label_setting: {setting}. Supported: {', '.join(LABEL_SETTINGS)}")
    return setting


def is_multi_label(cfg: dict | None) -> bool:
    return get_label_setting(cfg) == MULTI_LABEL


def normalize_pred_types(raw_types, pred_type: str, label_set: list[str],
                         resolve=None) -> list[str]:
    """
    Build the final multi-label prediction set: vocabulary-normalised,
    deduplicated, with pred_type first.  `resolve` maps a raw string to a
    vocabulary label (e.g. reasoner.resolve_label_to_vocab).
    """
    if isinstance(raw_types, str):
        raw_types = [raw_types]
    out: list[str] = []
    for t in [pred_type] + list(raw_types or []):
        if not t:
            continue
        t = resolve(str(t), label_set) if resolve else str(t).strip()
        if label_set and t not in label_set:
            continue
        if t not in out:
            out.append(t)
    return out
