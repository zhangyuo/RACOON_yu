# -*- coding: utf-8 -*-
"""
Keep the table of a RACOON prompt inside the context window of the LLM server (RAG4CTA addition).

RACOON puts the whole table (all columns, first 6 rows) into every prompt as "Comma-separated Values"; row_to_csv() writes it with
standard CSV quoting (the original `", ".join(cells)` left commas inside cells unquoted, which shifts the column count of the LLM). A few tables have very long cells (descriptions, reviews);
such a prompt exceeds the context length of the server (32,768 tokens for GLM-4-9B on the shared vLLM) and the call fails.

fit_rows() leaves a table unchanged when it fits. Only when it does not fit, every cell of THE PROMPT TEXT is cut to at most `cap`
characters, trying 4000, 2000, 1200, 800, 400, 200, 100, 60, 40, 25, 15 until the table fits. The entity linking and the Wikidata cache keep using
the original cells; only the text shown to the LLM is shortened. Tokens are counted exactly with the /tokenize endpoint of the
vLLM server (falls back to 2 characters per token when the server cannot be reached).
"""
from __future__ import annotations  # RACOON runs on Python 3.8

import json
import os
import urllib.request

CONTEXT = 32768          # context length of the server
ANSWER = 2000            # max_tokens of a call (utils.response)
SLACK = 7000             # instructions, hint, the model's first answer in the retry, the pruning prompt's type lists
CAPS = (4000, 2000, 1200, 800, 400, 200, 100, 60, 40, 25, 15)
CHECK_FROM_CHARS = 30000  # below this many characters a table cannot reach the budget (even at 2 characters per token)


def count_tokens(text: str) -> int:
    base = os.environ.get("VLLM_BASE_URL", "").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        req = urllib.request.Request(
            base + "/tokenize",
            data=json.dumps({"model": os.environ.get("VLLM_MODEL", ""), "prompt": text}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("VLLM_API_KEY", "")})
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(json.load(r)["count"])
    except Exception:
        return len(text) // 2


def cell_to_csv(cell: str) -> str:
    """Standard CSV quoting: a cell that contains a comma, a quote or a line break is wrapped in double quotes (quotes doubled)."""
    cell = cell.replace("\x00", "")  # NUL characters (they occur in some SOTAB cells) are not text and break CSV parsers
    if any(ch in cell for ch in ',"\n\r'):
        return '"' + cell.replace('"', '""') + '"'
    return cell


def row_to_csv(row) -> str:
    """One table row as the prompt shows it: cells separated by ", " like the original RACOON, but with the quoting above so that
    a comma INSIDE a cell does not look like a column separator."""
    return ", ".join(cell_to_csv(c) for c in row)


def table_to_text(rows) -> str:
    """The table as shown to the LLM. Style from the env var CSV_STYLE:
       original       cells joined by ", " with no quoting (the original RACOON code; DEFAULT, the baseline is run as in the original)
       quoted         standard CSV quoting of cells that contain a comma, quote or line break
       quoted_header  quoted + a first line "Column 1, Column 2, ..." so that the number of a column can be read off the table"""
    style = os.environ.get("CSV_STYLE", "original")
    if style == "original":
        return ",\n".join(", ".join(c.replace("\x00", "") for c in r) for r in rows)
    body = ",\n".join(row_to_csv(r) for r in rows)
    if style == "quoted_header":
        n = max((len(r) for r in rows), default=0)
        return ", ".join(f"Column {i + 1}" for i in range(n)) + ",\n" + body
    return body


def _csv(rows) -> str:
    return table_to_text(rows)


def fit_rows(rows, label_set):
    """rows: list of row tuples (cells as str). Returns (rows_for_the_prompt, info) where info is None when nothing was cut,
    else {"tokens_before", "tokens_after", "cap"}."""
    text = _csv(rows)
    if len(text) < CHECK_FROM_CHARS:
        return rows, None
    budget = CONTEXT - ANSWER - SLACK - count_tokens(str(list(label_set)))
    before = count_tokens(text)
    if before <= budget:
        return rows, None
    for cap in CAPS:
        cut = [tuple(c if len(c) <= cap else c[:cap] for c in r) for r in rows]
        after = count_tokens(_csv(cut))
        if after <= budget:
            return cut, {"tokens_before": before, "tokens_after": after, "cap": cap, "budget": budget}
    return cut, {"tokens_before": before, "tokens_after": after, "cap": cap, "budget": budget, "still_too_long": True}
