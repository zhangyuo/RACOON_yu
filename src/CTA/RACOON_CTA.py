import os
import sys
import csv
import json
from random import seed
from tqdm import tqdm
from pathlib import Path
# RAG4CTA fix: `.resolve()` is required -- when run as `python RACOON_CTA.py`
# (not an absolute path), `__file__` is relative, and pathlib's `.parent` on a
# bare relative name doesn't climb real directory levels, so `data_path`
# silently ended up wrong (pointing at ./data next to the script, not ../data).
data_path = Path(__file__).resolve().parent.parent / 'data'
sys.path.append(str(data_path))
from TURL_CTA_label_reduction import reduced_label_set
import openai
import re
import argparse

seed(0)
# NOTE(RAG4CTA): the actual model client now lives in utils.response(), pointed
# at our vLLM server -- this file no longer talks to the OpenAI API directly.

from refined.inference.processor import Refined
sys.path.append('..')
from KG_Linker import get_column_wise_spans
from pruning import *
from utils import *

# SC / DB tables come from src/sc_db_loader.py (reads the GitTables source fold files, see its docstring)
from sc_db_loader import load_tables as load_sc_db_tables, MAX_ROWS  # MAX_ROWS = 6: rows used per column

# RAG4CTA fix: see utils.py -- same data_dir override, home filesystem is nearly full.
refined = Refined.from_pretrained(model_name='wikipedia_model',
                                  entity_set="wikidata",
                                  data_dir=os.getenv("REFINED_CACHE_DIR", "/nfs/srv/data2/af62lp@cs.aau.dk/refined_cache"))
preprocessor = refined.preprocessor

def get_info_re(column):
    en_labels = []
    text = ""
    for (pid, em) in column:
        spans = refined.process_text(em)
        if (len(spans) < 1): continue
        item = spans[0]
        item_string = item.__repr__()
        res = item_string.strip("[]").split(", ")
        label_match = re.search(r'wikipedia_entity_title=([^,]+)', res[2][:-1])
        if label_match:
            entity = label_match.group(1)
            en_labels.append(entity)

    return en_labels


def get_col_entity(col_spans):
    en_labels = []
    for span in col_spans:
        if len(span) < 1: continue
        item = span[0] # for each cell, only take the first entity linked
        item_string = item.__repr__()
        res = item_string.strip("[]").split(", ")
        label_match = re.search(r'wikipedia_entity_title=([^,]+)', res[2][:-1])
        if label_match:
            entity = label_match.group(1)
            en_labels.append(entity)
    return en_labels

def main():
 
    
    parser = argparse.ArgumentParser(description='Column Type Annotation with RACOON+')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='OpenAI model to use (default: gpt-4o-mini; ignored for GLM, see utils.response)')
    parser.add_argument('--dataset', type=str, default='turl',
                        choices=['turl', 'sc', 'db'],
                        help='Dataset: turl (original) or sc/db (RAG4CTA GitTablesSC/DB, via sc_db_loader)')
    parser.add_argument('--context', type=str, default='col',
                        choices=['wikiAPI', 'cell', 'table', 'col', 'hybrid'],
                        help='Context type (default: hybrid)')
    parser.add_argument('--info', type=str, default='type',
                        choices=['entity', 'des', 'type', 'relation'],
                        help='Information type (default: type)')
    parser.add_argument('--max_tables', type=int, default=None,
                        help='RAG4CTA addition: cap the number of tables processed, for smoke tests')

    args = parser.parse_args()

    data_dir = os.path.join(os.getenv('DATA_DIR', '.'), 'data')
    model = args.model
    dataset = args.dataset
    context = args.context
    info = args.info

    if dataset != 'turl' and context == 'wikiAPI':
        raise NotImplementedError("wikiAPI context needs `colset`, which sc_db_loader does not build; "
                                   "use --context col/cell/table/hybrid for sc/db")

    # RAG4CTA fix (two issues):
    # 1. `--model` (default 'gpt-4o-mini') is a leftover OpenAI-era argument
    #    that no longer controls which model actually runs -- utils.response()
    #    always calls VLLM_MODEL -- so naming the output folder after it was
    #    actively misleading (results labelled "gpt-4o-mini" that were really
    #    GLM). Name the folder after the model that is actually running instead.
    # 2. Outputs were landing inside src/ (a source directory) because OUTPUT
    #    was a path relative to cwd. Write to <RACOON repo root>/outputs/ instead.
    model_label = os.environ.get('VLLM_MODEL', 'zai-org/GLM-4-9B-0414').replace('/', '-')
    racoon_root = Path(__file__).resolve().parent.parent.parent
    OUTPUT = str(racoon_root / 'outputs' / model_label /
                 (f"RACOON_{context}_{info}.csv" if dataset == 'turl'
                  else f"{dataset}/RACOON_{context}_{info}.csv"))
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    res_format = "{'type': []}"

    if dataset == 'turl':
        label_set = reduced_label_set
        # RAG4CTA fix: `data_dir` (os.getenv('DATA_DIR', '.') + 'data') is
        # relative to whatever cwd the script happens to be launched from --
        # same class of bug as the module-level `data_path` fixed above with
        # .resolve(). Use that already-correct absolute path instead.
        with open(data_path / 'test.table_col_type.json', 'r') as f:
            examples = json.load(f)
    else:
        examples, label_set = load_sc_db_tables(dataset)
        # RAG4CTA addition: skip tables that contain no evaluable column at
        # all (gold_type == '-1' for every column) -- they cost calls but can
        # never contribute to Micro/Macro F1.
        examples = [t for t in examples if any(g != '-1' for g in t['gold_types'])]
    if args.max_tables is not None:
        examples = examples[:args.max_tables]

    for id, example in enumerate(tqdm(examples)):
        if dataset == 'turl':
            table_raw, table_with_en, label, headers, meta_data, num_col, max_num_row, colset = parse_example(example)
            targets = list(range(num_col))
        else:
            table_raw, num_col, colset = example['table_raw'], example['num_col'], None
            # RAG4CTA addition: only ask the model about columns that are
            # actually scored; the rest of the table is still full context
            # (CSV_like below still uses all num_col columns).
            targets = [i for i in range(num_col) if example['gold_types'][i] != '-1']
        x = min(len(table_raw), MAX_ROWS)  # only the first MAX_ROWS (6) rows of every column, for every dataset
        y = len(table_raw[0])
        table = []
        all_preds = []
        hints = []
        for col in range(y):
            column = []
            for row in range(x):
                if table_raw[row][col] == None:
                    column.append("")
                else: column.append(table_raw[row][col])
            table.append(column)
        table_cols = table.copy()
        table = list(zip(*table))
        table_row = table.copy()
        table = map(lambda x: ", ".join(x), table)
        CSV_like = ",\n".join(table)

        # --------------------- KG-Linker --------------------- #
        if context == "wikiAPI":
            for col in colset:
                col_qids = get_qid_wk(col)
                EL_result_table.append(col_qids)
        else:
            column_wise_table_spans = get_column_wise_spans(context, table_row, table_cols)
            EL_result_table = []
            for column_wise_table_span in column_wise_table_spans:
                EL_result_col = process_EL_res(column_wise_table_span)
                EL_result_table.append(EL_result_col)

        # --------------------- KG-Explorer --------------------- #
        hint = ""
        if info == "entity":
            table_entities = get_entity_label(EL_result_table)
            # hint = f"Each cell may be linked to an entity in the Wikidata knowledge graph. When available, the entity labels corresponding to the cells in the first column are provided as a list: {col_info}."
            orig_table_hint = []
            for c in range(num_col):
                t = "Column " + str(c) + ": " + str(table_entities[c])
                orig_table_hint.append(t)
            pruned_table_hint = pruning_orig(CSV_like, orig_table_hint)
            pattern = r"-?\s*Column \d+:(.*?)(?:\n|$)"
            hint_strings = re.findall(pattern, pruned_table_hint)
            hint = f"The entity labels in the Wikidata knowledge graph corresponding to the cells in this column are provided as a list: {hint_strings[targets[0]]}."

        elif info == "type":
            types,_ = get_types(EL_result_table)
            orig_types = []
            for c in range(num_col):
                t = "Column " + str(c) + ": " + str(types[c])
                orig_types.append(t)
            pruned_hint = pruning_orig(CSV_like,orig_types)
            pattern = r"-?\s*Column \d+:(.*?)(?:\n|$)"
            dict_strings = re.findall(pattern, pruned_hint)
            hint_list = [safe_parse_dict(d) for d in dict_strings]
            # RAG4CTA: the first message may target any column in `targets`, not
            # always column 0 -- same length-guard the follow-up loop uses below.
            hint = (serialize_dict(hint_list[targets[0]]) if targets[0] < len(hint_list)
                    else "Entities in this column are instances of the following wikidata entities: ")

        elif info == "des":
            table_entities = get_entity_label_des(EL_result_table)
            orig_table_hint = []
            for c in range(num_col):
                t = "Column " + str(c) + ": " + str(table_entities[c])
                orig_table_hint.append(t)
            pruned_table_hint = pruning_orig(CSV_like, orig_table_hint)
            pattern = r"-?\s*Column \d+:(.*?)(?:\n|$)"
            hint_strings = re.findall(pattern, pruned_table_hint)
            hint = f"The entity labels with descriptions in the Wikidata knowledge graph corresponding to the cells in this column are provided as a list: {hint_strings[targets[0]]}."
        
        elif info == "relation":
            table_relations = get_entity_relation(EL_result_table)
            pruned_table_rel = pruning_rel(CSV_like, table_relations)
            hint = "This table's columns are related in Wikidata as follows: " + pruned_table_rel

        # --------------------- Generation --------------------- #
        # Paper form (RACOON, Wei et al., Appendix A): every column is annotated with its OWN prompt, which carries the
        # whole table, the label set and the KG hint of that column. Nothing is carried over from the other columns
        # (the released code kept one growing conversation per table, so every further column made the prompt longer).

        def column_hint(c):
            if info == "type":
                return (serialize_dict(hint_list[c]) if c < len(hint_list)
                        else "Entities in this column are instances of the following wikidata entities: ")
            if info == "entity":
                return f"The entity labels in the Wikidata knowledge graph corresponding to the cells in this column are provided as a list: {hint_strings[c]}."
            if info == "des":
                return f"The entity labels with descriptions in the Wikidata knowledge graph corresponding to the cells in this column are provided as a list: {hint_strings[c]}."
            return hint  # relation: one hint for the whole table

        def annotate_column(col, hint):
            messages = [{
                    "role": "system",
                    "content": f"""You are a data type analysis agent designed to annotate columns with semantic types. Your task is to:
            1. **Analyze** the given column data in the context of the entire table.
            2. **Consider** a Wikidata hint if given. 
            3. **Reason** carefully to select the best single type from the given list.
            4. **Output** ONLY in valid JSON format: {res_format}.

            Follow this format precisely:
            THOUGHT PROCESS: [Brief reasoning, mention if using/ignoring hint]
            FINAL ANSWER: [only the JSON output here]"""
                },
                {
                    "role": "user",
                    "content": f"""Consider this table with {num_col} columns given in Comma-separated Values format:
                        {CSV_like}
                        There are a list of {len(label_set)} valid types for each column: {label_set}. Your task is to choose only one type from the list to annotate Column {col+1}.
            Solve this task by following these steps:
            1. **Analyze the data in Column {col+1} in the context of the entire table**
            2. **Evaluate the Wikidata hint if not empty: {hint}**:
            - Use the hint to enhance your understanding of the column data.
            3. **Select the best type** from the given label set. 
            - Ensure the type best summarizes every cell in the column.
            - Ensure it aligns with the context of the whole table.
            - Verify your chosen type MUST be in the provided list.
            4. **Output**:
            - First, explain your reasoning (non-JSON).
            - Finally, output **ONLY** the JSON answer in this format: {res_format}.
            Example:
            THOUGHT: The data in this column consists of names of sports teams. The hint lists 'football_club (7 cells)', which confirms that the column is about sports teams. I select 'sports_team'.
            FINAL ANSWER: {{"type": ["sports_team"]}}"""}
            ]


            chatgpt_msg = response(messages, model)
            prediction = chatgpt_msg.content
            predicted_type = parse_pred(prediction)
            messages.append(dict(chatgpt_msg))
            col_prompt_tok, col_completion_tok = last_usage["prompt_tokens"], last_usage["completion_tokens"]

            if predicted_type not in label_set:
                messages.append({"role": "user",
                    "content": f"""Your prediction {predicted_type} is not in the given label set. Please try again to choose the closest label in the given set to annotate the first column. Choose only one valid type from the given list of types. Check that the type MUST MUST MUST be in the list. Give the answer in valid JSON format. DO NOT include any explanation or additional text."""})
                chatgpt_msg = response(messages, model)
                prediction = chatgpt_msg.content
                messages.append(dict(chatgpt_msg))
                predicted_type = parse_json_pred(prediction)
                col_prompt_tok += last_usage["prompt_tokens"]
                col_completion_tok += last_usage["completion_tokens"]

            return (predicted_type, prediction, col_prompt_tok, col_completion_tok)

        hints.clear()
        for col in targets:
            col_hint = column_hint(col)
            hints.append(col_hint)
            all_preds.append(annotate_column(col, col_hint))

        with open(OUTPUT, "a", newline="", encoding="UTF-8") as output:
            writer = csv.writer(output)
            # `all_preds`/`hints` are in the order of `targets`, not 0..num_col-1
            # (for sc/db, non-evaluable columns were skipped entirely).
            for pos, (col_idx, p) in enumerate(zip(targets, all_preds)):
                if dataset == 'turl':
                    writer.writerow([id, pos, p[0], hints[pos], p[1], p[2], p[3]])
                else:
                    # RAG4CTA addition: carry table_id/col_id/gold_type, and
                    # prompt/completion token counts (p[2]/p[3], read straight
                    # off the vLLM response's own `usage` field -- see
                    # utils.response()/last_usage), so the output is
                    # self-contained for src/eval/metrics.py and cost analysis,
                    # no separate id-join file needed.
                    writer.writerow([id, pos, p[0], hints[pos], p[1], p[2], p[3],
                                      example['table_id'], example['col_ids'][col_idx], example['gold_types'][col_idx]])

if __name__ == "__main__":
    main()