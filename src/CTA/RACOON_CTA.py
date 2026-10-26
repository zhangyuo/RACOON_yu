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

# SC / DB tables come from src/sc_db_loader.py (reads data/<dataset>/{train,valid,test}.csv, see its docstring)
from sc_db_loader import load_tables as load_sc_db_tables, MAX_ROWS  # MAX_ROWS = 6: rows used per column
from label_resolve import resolve_to_label_set
from context_fit import fit_rows, table_to_text

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
                        choices=['turl', 'sc', 'db', 'biodiv_db', 'biodiv_wiki', 'sotab'],
                        help='Dataset: turl (original) or sc/db/biodiv_db/biodiv_wiki (RAG4CTA datasets, via sc_db_loader)')
    parser.add_argument('--context', type=str, default='col',
                        choices=['wikiAPI', 'cell', 'table', 'col', 'hybrid'],
                        help='Context type (default: hybrid)')
    parser.add_argument('--info', type=str, default='type',
                        choices=['entity', 'des', 'type', 'relation'],
                        help='Information type (default: type)')
    parser.add_argument('--max_tables', type=int, default=None,
                        help='RAG4CTA addition: cap the number of tables processed, for smoke tests')
    parser.add_argument('--label_setting', type=str, default='single_label',
                        choices=['single_label', 'multi_label'],
                        help='RAG4CTA addition: single_label = the model picks ONE type per column (the prompt of the paper); '
                             'multi_label = the model lists ALL types that apply (same prompt, only the selection rule '
                             'and the example change). The two settings are separate runs with separate result files.')
    parser.add_argument('--label_space', type=str, default='freebase', choices=['freebase', 'reduced'],
                        help="RAG4CTA addition, TURL only: freebase (default) = the 255 Freebase type names of the benchmark "
                             "(type_vocab.txt), the label set of the other methods; reduced = the 127 simplified type names of the "
                             "original RACOON code. Results go to outputs/<model>/turl_<label_space>/; the freebase files carry "
                             "table_id/col_id/gold columns.")
    parser.add_argument('--only', type=str, default=None,
                        help="RAG4CTA debugging aid: TABLE_ID:COLUMN_INDEX,... run only these columns (sc/db/biodiv datasets); "
                             "the output file gets the tag .only")
    parser.add_argument('--only_file', type=str, default=None,
                        help="RAG4CTA addition: like --only, but the columns are read from a tab separated file with the lines "
                             "TABLE_ID<TAB>COLUMN_INDEX (any dataset incl. turl, where the column index is the position in the table)")
    parser.add_argument('--dump_dialog', type=str, default=None,
                        help="RAG4CTA debugging aid: append the complete conversation of every asked column (system prompt, user "
                             "prompt, answer, retry message, retry answer) as one JSON line to this file")
    parser.add_argument('--shard', type=str, default=None,
                        help='RAG4CTA addition: K/N, process only the tables with index %% N == K (parallel runs). '
                             'Every shard writes its own file RACOON_..._shardKofN.csv; concatenate them afterwards.')

    args = parser.parse_args()

    data_dir = os.path.join(os.getenv('DATA_DIR', '.'), 'data')
    model = args.model
    dataset = args.dataset
    context = args.context
    info = args.info
    multi = args.label_setting == 'multi_label'

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
    shard_k, shard_n = (int(x) for x in args.shard.split('/')) if args.shard else (0, 1)
    shard_tag = (".rerun" if (args.only or args.only_file) else "") + (f".shard{shard_k}of{shard_n}" if args.shard else "")
    out_name = f'turl_{args.label_space}' if dataset == 'turl' else dataset
    OUTPUT = str(racoon_root / 'outputs' / model_label / out_name / args.label_setting / os.environ.get('RUN_TAG', '') /
                 f"RACOON_{context}_{info}{shard_tag}.csv")
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    res_format = "{'type': []}"

    if dataset == 'turl':
        if args.label_space == 'freebase':
            # the 255 Freebase names of the benchmark, the same label set as the other methods (type_vocab.txt of the shared data)
            from sc_db_loader import ROOT as DATA_ROOT
            label_set = [l.rstrip("\n").split("\t")[1].strip() for l in
                         open(DATA_ROOT / "data" / "TURL-Wiki-CTA" / "type_vocab.txt", encoding="utf-8") if "\t" in l]
        else:
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
    only_cols = {(t, int(c)) for t, c in (x.rsplit(':', 1) for x in args.only.split(','))} if args.only else None
    if args.only_file:
        only_cols = set()
        for line in open(args.only_file, encoding="utf-8"):
            if line.strip():
                t, c = line.rstrip("\n").split("\t")
                only_cols.add((t, int(c)))
    if only_cols:
        wanted = {t for t, _ in only_cols}
        examples = [e for e in examples if (e[0] if dataset == 'turl' else e['table_id']) in wanted]
    # `id` stays the index of the table in the full list, so the rows of all shards can be concatenated
    indexed = [(i, e) for i, e in enumerate(examples) if i % shard_n == shard_k]

    for id, example in tqdm(indexed):
        if dataset == 'turl':
            table_raw, table_with_en, label, headers, meta_data, num_col, max_num_row, colset = parse_example(example)
            targets = list(range(num_col))
            if only_cols:
                targets = [i for i in targets if (example[0], i) in only_cols]
        else:
            table_raw, num_col, colset = example['table_raw'], example['num_col'], None
            # RAG4CTA addition: only ask the model about columns that are
            # actually scored; the rest of the table is still full context
            # (CSV_like below still uses all num_col columns).
            targets = [i for i in range(num_col) if example['gold_types'][i] != '-1']
            if only_cols:
                targets = [i for i in targets if (example['table_id'], example['col_ids'][i]) in only_cols]
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
        # RAG4CTA addition: a table that does not fit into the context of the server is shown to the LLM with shortened cells
        # (the linking below keeps using the original cells); tables that fit are not touched
        table_llm, fit_info = fit_rows(table_row, label_set)
        if fit_info:
            tid = example[0] if dataset == 'turl' else example['table_id']
            with open(OUTPUT + ".truncated.jsonl", "a", encoding="utf-8") as tf:
                tf.write(json.dumps({"table_id": tid, **fit_info}) + "\n")
        CSV_like = table_to_text(table_llm)  # RAG4CTA: standard CSV quoting (env CSV_STYLE), see context_fit.py

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

        # Selection rule of the prompt. single_label is the prompt of the code/paper, unchanged. multi_label follows the
        # multi-label prompt of the paper (Appendix A.2): "choose one or multiple types", step "Mark each type in the given
        # list with 0 or 1 ...", step "Give a list of types that you have marked 1 ...". Three places are not given by the
        # paper (the code's own system-prompt line, its worked example, its retry message) and reuse the paper's wording.
        if multi:
            w_sys = "choose one or multiple types from the given list"
            w_task = "choose one or multiple types from the list"
            w_step3 = ("Mark each type in the given list with 0 or 1. Mark a type with 1 if it can better represent all cells "
                       "of Column {c}. Mark a type with 0 otherwise.")
            w_ens1 = ""
            w_ens3 = ""
            w_out = ("\n            - Give a list of types that you have marked 1 in the previous step. "
                     "Check that the types MUST be in the list.")
            w_example = ('THOUGHT: The data in this column consists of names of sports teams. The hint lists \'football_club (7 cells)\', '
                         'which confirms that the column is about sports teams, and a sports team is also an organization. '
                         'I mark \'sports_team\' and \'organization\' with 1 and every other type with 0.\n            '
                         'FINAL ANSWER: {"type": ["sports_team", "organization"]}')
            w_retry = "Choose one or multiple valid types from the given list of types. Check that the types MUST MUST MUST be in the list."
        else:
            w_sys = "select the best single type from the given list"
            w_task = "choose only one type from the list"
            w_step3 = "**Select the best type** from the given label set. "
            w_ens1 = "\n            - Ensure the type best summarizes every cell in the column."
            w_ens3 = "\n            - Verify your chosen type MUST be in the provided list."
            w_out = ""
            w_example = ('THOUGHT: The data in this column consists of names of sports teams. The hint lists \'football_club (7 cells)\', '
                         'which confirms that the column is about sports teams. I select \'sports_team\'.\n            '
                         'FINAL ANSWER: {"type": ["sports_team"]}')
            w_retry = "Choose only one valid type from the given list of types. Check that the type MUST MUST MUST be in the list."

        def annotate_column(col, hint):
            messages = [{
                    "role": "system",
                    "content": f"""You are a data type analysis agent designed to annotate columns with semantic types. Your task is to:
            1. **Analyze** the given column data in the context of the entire table.
            2. **Consider** a Wikidata hint if given. 
            3. **Reason** carefully to {w_sys}.
            4. **Output** ONLY in valid JSON format: {res_format}.

            Follow this format precisely:
            THOUGHT PROCESS: [Brief reasoning, mention if using/ignoring hint]
            FINAL ANSWER: [only the JSON output here]"""
                },
                {
                    "role": "user",
                    "content": f"""Consider this table with {num_col} columns given in Comma-separated Values format:
                        {CSV_like}
                        There are a list of {len(label_set)} valid types for each column: {label_set}. Your task is to {w_task} to annotate Column {col+1}.
            Solve this task by following these steps:
            1. **Analyze the data in Column {col+1} in the context of the entire table**
            2. **Evaluate the Wikidata hint if not empty: {hint}**:
            - Use the hint to enhance your understanding of the column data.
            3. {w_step3.format(c=col+1)}{w_ens1}
            - Ensure it aligns with the context of the whole table.{w_ens3}
            4. **Output**:
            - First, explain your reasoning (non-JSON).{w_out}
            - Finally, output **ONLY** the JSON answer in this format: {res_format}.
            Example:
            {w_example}"""}
            ]


            chatgpt_msg = response(messages, model)
            prediction = chatgpt_msg.content
            predicted_type = resolve_to_label_set(parse_pred_list(prediction) if multi else parse_pred(prediction), label_set)
            messages.append(dict(chatgpt_msg))
            col_prompt_tok, col_completion_tok = last_usage["prompt_tokens"], last_usage["completion_tokens"]

            if (not predicted_type or any(t not in label_set for t in predicted_type)) if multi else predicted_type not in label_set:
                messages.append({"role": "user",
                    "content": f"""Your prediction {predicted_type} is not in the given label set. Please try again to choose the closest label in the given set to annotate the first column. {w_retry} Give the answer in valid JSON format. DO NOT include any explanation or additional text."""})
                chatgpt_msg = response(messages, model)
                prediction = chatgpt_msg.content
                messages.append(dict(chatgpt_msg))
                predicted_type = resolve_to_label_set(parse_json_pred_list(prediction) if multi else parse_json_pred(prediction), label_set)
                col_prompt_tok += last_usage["prompt_tokens"]
                col_completion_tok += last_usage["completion_tokens"]

            if args.dump_dialog:
                with open(args.dump_dialog, "a", encoding="utf-8") as dump:
                    dump.write(json.dumps({"table_id": example['table_id'], "col_id": example['col_ids'][col], "hint": hint,
                                           "messages": messages}, ensure_ascii=False, default=str) + "\n")
            if multi:  # the vocabulary types only, deduplicated, in the order given; stored as 'a;b;c'
                predicted_type = ";".join(dict.fromkeys(t for t in predicted_type if t in label_set))
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
                if dataset == 'turl' and args.label_space == 'freebase':
                    # table_id / column position / gold (Freebase names joined by ';' in gold order), like the other datasets
                    writer.writerow([id, pos, p[0], hints[pos], p[1], p[2], p[3],
                                      example[0], col_idx, ";".join(label[col_idx])])
                elif dataset == 'turl':
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