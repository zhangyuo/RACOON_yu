import os
import re
from utils import compute_cell_boundaries
from refined.inference.processor import Refined

# RAG4CTA fix: see utils.py -- same data_dir override, home filesystem is nearly full.
refined = Refined.from_pretrained(model_name='wikipedia_model',
                                  entity_set="wikidata",
                                  data_dir=os.getenv("REFINED_CACHE_DIR", "/nfs/srv/data2/af62lp@cs.aau.dk/refined_cache"))


def _group_spans_by_cell(text, spans):
    """
    RAG4CTA replacement for RACOON's `refined.process_text(..., cell_boundaries=...)`
    call: that keyword argument does not exist in any public release of ReFinED
    (checked the tagged V1 release and the current main branch) -- RACOON's
    authors must have used an unpublished fork. This reimplements the same
    "which cell does each entity belong to" grouping using only what the
    public ReFinED returns: each Span's character start/length in `text`.

    `text` is always built elsewhere as "|| " + " || ".join(cells) + " ||"
    (cell/col/table context) or without the trailing "||" (hybrid context).
    Splitting on the literal "||" delimiter and dropping the empty leading/
    trailing wrapper segments recovers exactly one segment per real cell, in
    the same left-to-right order the old token-boundary version produced.
    """
    delims = [(m.start(), m.end()) for m in re.finditer(r'\|\|', text)]
    seg_starts = [0] + [e for _, e in delims]
    seg_ends = [s for s, _ in delims] + [len(text)]
    cell_segments = [(s, e) for s, e in zip(seg_starts, seg_ends) if text[s:e].strip()]

    buckets = [[] for _ in cell_segments]
    for span in spans:
        for i, (s, e) in enumerate(cell_segments):
            if s <= span.start < e:
                buckets[i].append(span)
                break
    return buckets

def get_column_wise_spans(context, table, table_cols):
    """
    Perform entity linking on table data using different context types.
    
    Args:
        context: The context type for entity linking, one of:
                - 'cell': Each cell processed independently
                - 'col': Entire column processed together
                - 'table': Entire table processed together (column-wise)
                - 'hybrid': Row + column hybrid context
        table: Row-wise representation of the table (list of rows)
        table_cols: Column-wise representation of the table (list of columns)
    
    Returns:
        column_wise_table_spans: A list where each element corresponds to a column,
                                and each column contains a list of entity spans for
                                each cell in that column.
    """
    column_wise_table_spans = []
    
    if context == 'cell':
        # Process each cell independently
        for col in table_cols:
            col_spans = []
            for cell in col:
                text = "|| " + (cell if cell.strip() else "-") + " ||"
                spans = refined.process_text(text=text)
                cell_sep_spans = _group_spans_by_cell(text, spans)
                if len(cell_sep_spans) > 0:
                    col_spans.append(cell_sep_spans[0])
                else:
                    col_spans.append([])
            column_wise_table_spans.append(col_spans)
    
    elif context == 'col':
        # Process each column as a whole
        for col in table_cols:
            text = "|| " + " || ".join(item if item.strip() else "-" for item in col) + " ||"
            spans = refined.process_text(text=text)
            cell_sep_spans = _group_spans_by_cell(text, spans)
            if len(cell_sep_spans) == 0:
                cell_sep_spans = [[] for _ in range(len(col))]
            column_wise_table_spans.append(cell_sep_spans)
    
    elif context == 'table':
        # Process entire table at once (column-wise ordering)
        table_text = "|| " + " || ".join(
            " || ".join(item if item.strip() else "-" for item in col) for col in table_cols
        ) + " ||"
        
        spans = refined.process_text(text=table_text)
        cell_sep_spans = _group_spans_by_cell(table_text, spans)
        
        for col_idx in range(len(table_cols)):
            col_spans = []
            length = len(table_cols[col_idx])
            for row_idx in range(length):
                flat_idx = col_idx * length + row_idx
                if flat_idx < len(cell_sep_spans):
                    col_spans.append(cell_sep_spans[flat_idx])
                else:
                    col_spans.append([])
            column_wise_table_spans.append(col_spans)
    
    elif context == 'hybrid':
        # Process with row + column hybrid context
        table_spans = []
        for n, row in enumerate(table):
            row_spans = []
            for cid, cell in enumerate(row):
                linked_en = []
                text = "|| " + " || ".join(item if item.strip() else "-" for item in row) + \
                       " || " + " || ".join(table[r][cid] if table[r][cid].strip() else "-" for r in range(len(table)))
                spans = refined.process_text(text=text)
                cell_sep_spans = _group_spans_by_cell(text, spans)
                if len(cell_sep_spans) != 0:
                    linked_en = cell_sep_spans[cid]
                row_spans.append(linked_en)
            table_spans.append(row_spans)
        column_wise_table_spans = list(zip(*table_spans))
    
    return column_wise_table_spans
