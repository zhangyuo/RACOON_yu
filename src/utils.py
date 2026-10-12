import numpy as np
import os
import csv
import json
import time
import subprocess
from sklearn.metrics import multilabel_confusion_matrix
import sys
from SPARQLWrapper import SPARQLWrapper, JSON
import openai
import ast
import re
from collections import defaultdict, Counter
from wikidata_api import fetch_entities
data_dir = os.getenv('DATA_DIR', './data')

# RAG4CTA addition: Wikidata's public SPARQL endpoint (query.wikidata.org) currently
# rate-limits to 1 req/min during an active WDQS outage, so every live query below
# mostly fails -- cache every *successful* lookup on disk, in ONE shared file with
# a section per query shape, so nothing already-successful is ever re-queried
# (failures are not cached, since they may succeed once the outage clears).
# "entity_full" merges what used to be get_instance_of's and get_entity_info's
# separate queries (both take a single QID) into one combined query/cache entry --
# see _fetch_entity_full(). "relation" stays separate since it takes a QID *pair*.
_CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
_WIKIDATA_CACHE_PATH = os.path.join(_CACHE_DIR, "wikidata_cache.json")

def _load_wikidata_cache():
    if os.path.exists(_WIKIDATA_CACHE_PATH):
        try:
            with open(_WIKIDATA_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    else:
        data = {}
    # Back-compat: migrate the older split "instance_of"/"entity_info" sections
    # (from before the two queries were merged) into "entity_full" if present.
    if "instance_of" in data or "entity_info" in data:
        merged = data.setdefault("entity_full", {})
        for qid, v in data.pop("instance_of", {}).items():
            merged.setdefault(qid, {}).update({
                "qids": v["qids"], "entity_set": v["entity_set"],
                "raw_bindings": v["raw_bindings"],
                "label": merged.get(qid, {}).get("label", ""),
                "description": merged.get(qid, {}).get("description", ""),
            })
        for qid, v in data.pop("entity_info", {}).items():
            merged.setdefault(qid, {}).update(v)
        with open(_WIKIDATA_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)  # persist the migration immediately
    data.setdefault("entity_full", {})  # get_instance_of() + get_entity_info() -- entity/type/des modes
    data.setdefault("relation", {})     # get_relation_single_direction()      -- relation mode
    return data

_wikidata_cache = _load_wikidata_cache()
_entity_full_cache = _wikidata_cache["entity_full"]   # kept as a local alias, same dict object
_relation_cache = _wikidata_cache["relation"]         # kept as a local alias, same dict object

def _save_wikidata_cache():
    with open(_WIKIDATA_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_wikidata_cache, f, ensure_ascii=False)

# kept as a thin alias so call sites below don't need to change name
_save_entity_full_cache = _save_wikidata_cache
_save_relation_cache = _save_wikidata_cache

from refined.inference.processor import Refined

# RAG4CTA fix: default cache dir (~/.cache/refined) is on a nearly-full home
# filesystem; point at the NFS drive instead (see also KG_Linker.py, CTA/RACOON_CTA.py).
refined = Refined.from_pretrained(model_name='wikipedia_model',
                                  entity_set="wikidata",
                                  data_dir=os.getenv("REFINED_CACHE_DIR", "/nfs/srv/data2/af62lp@cs.aau.dk/refined_cache"))
preprocessor = refined.preprocessor

def parse_example(example):
    label = example[7]
    meta_data = {}
    meta_data["page title"] = example[1]
    meta_data["section_title"] = example[3]
    meta_data["caption"] = example[4]
    headers = example[5]
    num_col = len(example[6])
    colset = []
    max_num_row = -1

    # build up a 2-D array for the table
    for i in range(num_col):
        col = example[6][i]

        num_row = len(col)
        table_col = []
        for j in range(num_row):
            cell = col[j]
            idx = cell[0]
            text = cell[1][1]
            given_link = cell[1][0]
            if idx[0] > max_num_row:
                max_num_row = idx[0]
            table_col.append((given_link, text))
        colset.append(table_col)

    max_num_row += 1
    table = np.empty([max_num_row, num_col], dtype = object) 
    table_with_en = np.empty([max_num_row, num_col], dtype = object) 

    for i in range(num_col):
        col = example[6][i]

        num_row = len(col)
        for j in range(num_row):
            cell = col[j]
            idx = cell[0]
            text = cell[1][1]
            page_id = cell[1][0]
            table[idx[0]][idx[1]] = text
            table_with_en[idx[0]][idx[1]] = (page_id, text)
            
    return table, table_with_en, label, headers, meta_data, num_col, max_num_row, colset


# RAG4CTA adaptation: point at our self-hosted vLLM server (OpenAI-compatible
# endpoint) instead of the official OpenAI API. `model` argument is kept for
# call-site compatibility but ignored -- the vLLM server only serves one model.
_vllm_client = openai.OpenAI(
    api_key=os.environ["VLLM_API_KEY"],
    base_url=os.environ.get("VLLM_BASE_URL", "http://130.225.39.64:8000/v1"),
)
_VLLM_MODEL = os.environ.get("VLLM_MODEL", "zai-org/GLM-4-9B-0414")


# RAG4CTA addition: after each response() call, this dict (mutated in place,
# so `from utils import *` star-imports still see updates) holds
# {"prompt_tokens", "completion_tokens"} for that call, read straight from the
# vLLM server's own OpenAI-compatible response -- no re-tokenizing/estimating.
# Kept as a side channel (not a second return value) so NoHint_baseline.py /
# SelfHint_baseline.py / RACOON_RE.py, which also call response() and expect a
# single return value, keep working unchanged.
last_usage = {"prompt_tokens": 0, "completion_tokens": 0}


def response(messages, model):
    # RAG4CTA fix: was calling with no temperature/seed at all, so every call
    # sampled with vLLM's server default and no fixed seed -- two runs of
    # identical code on identical data gave different F1 (measured: SC .4347
    # vs .4133). Fix temperature=0 and seed=42 (matching RAG4CTA's own
    # LLM4CTA baseline, src/llm/adapter.py's default config) so a
    # RACOON-vs-LLM4CTA comparison isn't also secretly a
    # determinism-vs-randomness comparison.
    #
    # RAG4CTA fix: RACOON keeps one growing conversation across every column
    # of a table, and each turn's reply gets appended to the next turn's
    # prompt. Measured on a real TURL table: completion length grows turn
    # over turn (183 -> 228 -> 345 -> 488 tokens) with no ceiling, and a
    # table with enough columns eventually produces a single call slow
    # enough to exceed the client's timeout (openai.APITimeoutError,
    # reproduced twice on the TURL smoke test). max_tokens=2000 is generous
    # (SC/DB's own completions average ~135 tokens) but stops the runaway
    # case; the retry loop below covers any timeout that still slips through.
    for attempt in range(3):
        try:
            completion = _vllm_client.chat.completions.create(
                model=_VLLM_MODEL,
                messages=messages,
                temperature=0,
                seed=42,
                max_tokens=2000,
            )
            break
        except openai.APITimeoutError:
            if attempt == 2:
                raise
            print(f"[utils.response] timeout, retrying ({attempt + 1}/3)", flush=True)
    msg = completion.choices[0].message
    # RAG4CTA fix: this GLM/vLLM deployment sometimes puts the actual reply in
    # `reasoning` and leaves `content` empty (see RAG4CTA's own
    # scripts/exp_tree_descent.py, which has the same fallback). RACOON_CTA.py
    # only ever reads `.content`, so backfill it here, once, in one place.
    if not msg.content:
        msg.content = getattr(msg, "reasoning", None) or ""
    if completion.usage:
        last_usage["prompt_tokens"] = completion.usage.prompt_tokens
        last_usage["completion_tokens"] = completion.usage.completion_tokens
    else:
        last_usage["prompt_tokens"] = 0
        last_usage["completion_tokens"] = 0
    return msg

def _fetch_entity_full(qid):
    """
    RAG4CTA: merged version of what used to be two separate live queries
    (get_instance_of's P31 "instance of" lookup and get_entity_info's
    label+description lookup) -- both only need a single QID, so they are
    combined into ONE SPARQL request per QID, halving the number of live
    calls against Wikidata's rate-limited endpoint. get_relation_single_direction
    is NOT folded in here: it takes a *pair* of QIDs, a different query shape.

    Returns a dict (and caches it under _entity_full_cache[qid]) with:
        label, description       -- from get_entity_info's old query
        qids, entity_set, raw_bindings  -- from get_instance_of's old query
    or None on failure (not cached, so a later call can retry).
    """
    if qid in _entity_full_cache:
        return _entity_full_cache[qid]

    # RAG4CTA: one batched Action-API request (wikidata_api.py) instead of a SPARQL query on
    # query.wikidata.org, which is rate limited to 1 request/minute. Same record format plus extra fields.
    try:
        result = fetch_entities([qid]).get(qid)
    except Exception:
        return None
    if result is None:
        return None
    _entity_full_cache[qid] = result
    _save_entity_full_cache()
    return result

def get_instance_of(qid):
    """
    Retrieve entity types (P31: instance of) for a given Wikidata QID.
    Backed by the merged _fetch_entity_full() query/cache (see its docstring).

    Returns:
        qids: List of type QIDs found for this entity
        entity_set: Set of human-readable labels for the types
    """
    result = _fetch_entity_full(qid)
    if result is None:
        return [], set()
    return result["qids"], set(result["entity_set"])

def get_entity_info(qid):
    """
    Retrieve the label and description for a Wikidata entity.
    Backed by the merged _fetch_entity_full() query/cache (see its docstring).

    Returns:
        dict: {'label': ..., 'description': ...}
        None: if the query fails or the entity is not found
    """
    result = _fetch_entity_full(qid)
    if result is None:
        return None
    return {"label": result["label"], "description": result["description"]}

def get_relation_single_direction(qid1, qid2):
    cache_key = f"{qid1}|{qid2}"
    if cache_key in _relation_cache:
        return set(_relation_cache[cache_key]["rel_set"])

    sparql = SPARQLWrapper("http://query.wikidata.org/sparql")
    sparql.setQuery(f"""SELECT ?property ?propertyLabel WHERE {{
        VALUES (?entity1 ?entity2) {{(wd:{qid1} wd:{qid2})}}

        {{
            ?entity1 ?property ?entity2.
        }}

        ?propertyEntity wikibase:directClaim ?property.
        ?propertyEntity rdfs:label ?propertyLabel.
        FILTER(LANG(?propertyLabel) = "en")
        }}
        """)
    sparql.setReturnFormat(JSON)
    rel_set = set()
    raw_bindings = []
    try :
        ret = sparql.query()
        results = ret.convert()
        data = results["results"]["bindings"]
    except :
        return rel_set

    for binding in data:
        label = binding['propertyLabel']['value']
        rel_set.add(label)
        # RAG4CTA: `property` is the raw property URI used in the triple (e.g.
        # .../prop/direct/P17) -- keep its QID-style suffix too, not just the
        # label, so the cache preserves everything Wikidata actually returned.
        raw_bindings.append({
            "property_id": binding['property']['value'].split('/')[-1],
            "property_label": label,
        })

    if rel_set:
        _relation_cache[cache_key] = {"rel_set": list(rel_set), "raw_bindings": raw_bindings}
        _save_relation_cache()

    return rel_set
    
def curl_request(url):
    
    # Define the command to execute using curl
    command = ['curl', '-s', '-o', '-', url]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout

def wikidata_lookup(em):
    em = em.replace(" ", "_")
    url = f'https://www.wikidata.org/w/api.php?action=wbsearchentities&search={em}&format=json&language=en&uselang=en&type=item&limit=1'
    response = curl_request(url)
    qid = None
    entity = None
    if response:  # Check if the response is not empty
        try:
            data = json.loads(response)
            content = data.get('search', [])
            if content:
                item = content[0]
                qid = item.get('id', {})
                entity = item.get('label', {})

        except json.JSONDecodeError as e:
            return qid, entity
    return qid, entity

def get_qid_wk(column):
    col_qids = []
    for (pid, em) in column:
      qid, entity_label = wikidata_lookup(em)
      if qid is None: qid = ''
      print(qid)
      col_qids.append(qid)
    return col_qids

def get_types(EL_res):
    """
    Extract and aggregate entity types for each column in a table.
    
    This function processes entity linking results at the table level, where each column's
    entities are analyzed to determine their types. For cells with multiple possible types,
    the most frequent type across the column is selected.
    
    Args:
        EL_res: A 2D array where each element represents a column's entity linking results.
                Each column contains a list of QIDs (Wikidata identifiers) for its cells.
    
    Returns:
        types: A list of dictionaries (one per column), where each dictionary maps
               type QIDs to the number of cell indices that belong to that type.
        col_type_size: A dictionary mapping column indices to the number of distinct
                       types found in that column.
    """
    col_type_size = {}
    types = []  # List of type counters for each column

    for i in range(len(EL_res)):  # Iterate through each column
        counter = Counter()
        col_qids = EL_res[i]
        cell_to_types_set = {}

        # First pass: collect all types for each cell and count type frequencies
        for j, qid in enumerate(col_qids):  # Iterate through each cell
            type_qids, entity_set = get_instance_of(qid)
            cell_to_types_set[j] = entity_set
            if len(entity_set) < 1: 
                continue
            counter.update(entity_set)

        # Second pass: assign the most appropriate type to each cell
        final_counter = {}
        for cell, types_set in cell_to_types_set.items():
            types_set = list(types_set)
            if len(types_set) < 1: 
                continue
            elif len(types_set) == 1:
                # Single type: assign directly
                t = types_set[0]
                if t not in final_counter: 
                    final_counter[t] = []
                final_counter[t].append(cell)
            else:
                # Multiple types: select the most frequent type in the column
                max_count = 0
                max_type = None
                for t in types_set:
                    count = counter[t]
                    if count > max_count:
                        max_count = count
                        max_type = t
                if max_type not in final_counter: 
                    final_counter[max_type] = []
                final_counter[max_type].append(cell)
        type_dict = {key: len(value) for key, value in final_counter.items()}
        types.append(type_dict)
        size = len(final_counter)
        col_type_size[i] = size
    
    return types, col_type_size

def get_entity_label(EL_res):
    """
    Extract human-readable entity labels for all cells in a table.
    
    This is a table-wise function that converts entity linking results (QIDs)
    into their corresponding Wikidata labels. For each column in the table,
    it retrieves the entity label for every cell that has been linked to a
    Wikidata entity.
    
    Args:
        EL_res: A 2D array where each element represents a column's entity linking
                results. Each column contains a list of QIDs corresponding to the cells in that column.
    
    Returns:
        table_res: A 2D array with the same structure as EL_res, but with QIDs
                   replaced by their human-readable labels.
    
    """
    table_res = []
    for i in range(len(EL_res)):  # For each column
        entity_labels = []
        col_qids = EL_res[i]

        # Loop through all cells in the column to get entity labels
        for j, qid in enumerate(col_qids):  # For each cell
            res_dict = get_entity_info(qid)
            entity_label = res_dict.get("label") if res_dict else None
            entity_labels.append(entity_label)

        table_res.append(entity_labels)
    
    return table_res

def get_entity_label_des(EL_res):
    """
    Extract entity labels and descriptions for all cells in a table.
    
    This is a table-wise function that converts entity linking results (QIDs)
    into their corresponding Wikidata labels and descriptions. For each column
    in the table, it retrieves both the entity label and description for every
    cell that has been linked to a Wikidata entity, combining them in the format
    "label: description".
    
    Args:
        EL_res: A 2D array where each element represents a column's entity linking
                results. Each column contains a list of QIDs (Wikidata identifiers)
                corresponding to the cells in that column.
    
    Returns:
        table_res: A 2D array with the same structure as EL_res, but with QIDs
                   replaced by formatted strings combining labels and descriptions.
    """
    table_res = []
    for i in range(len(EL_res)):  # For each column
        entity_info = []
        col_qids = EL_res[i]

        # Loop through all cells in the column to get entity labels and descriptions
        for j, qid in enumerate(col_qids):  # For each cell
            res_dict = get_entity_info(qid)
            if res_dict is None:
                entity_info.append("")
            else: 
                entity_label = res_dict.get("label")
                entity_des = res_dict.get("description")
                entity_info.append(entity_label + ": " + entity_des)

        table_res.append(entity_info)
    
    return table_res

def RE_get_col_relations(column1, column2):
    rel_set = Counter()
    length = min(len(column1), len(column2))
    for i in range(length):
        qid1 = column1[i]
        qid2 = column2[i]
        if qid1 == "" or qid2 == "": continue
        cur_set = get_relation_single_direction(qid1, qid2)
        rel_set.update(cur_set)
    return rel_set

def serialize_counter(counter):
    serialized_str = ""
    for item, count in counter.items():
        serialized_str += f"{item} ({count} cells), "

    return serialized_str.rstrip(", ")

def get_entity_relation(EL_res):
    """
    Extract relationships between the first column and all other columns in a table.
    
    This is a table-wise function that identifies Wikidata relationships (properties)
    between entities in the first column and entities in each subsequent column.
    
    Args:
        EL_res: A 2D array where each element represents a column's entity linking
                results. Each column contains a list of QIDs (Wikidata identifiers)
                corresponding to the cells in that column.
    
    Returns:
        table_res: A list of strings, one for each column (excluding the first column).
                   Each string describes the relationships found between the first column
                   and that column, formatted as:
                   "- Column 1 & Column {i+1}: relationship1 (count), relationship2 (count), ..."
    """
    table_res = []
    first_col = EL_res[0]
    for i in range(1, len(EL_res)):  # For each column (starting from the second)
        col_qids = EL_res[i]
        rel_set = RE_get_col_relations(first_col, col_qids)
        rel_str = f"- Column 1 & Column {i+1}: " + serialize_counter(rel_set)
        table_res.append(rel_str)
    
    return table_res


def serialize_dict(data):
    serialized_str = "Entities in this column are instances of the following wikidata entities: "
    for key, value in data.items():
        count = value
        cell_word = "cell" if count == 1 else "cells"
        serialized_str += f"{key} ({count} {cell_word}), "
    
    serialized_str = serialized_str.rstrip(", ")
    return serialized_str

def compute_cell_boundaries(flattened_table: str):
    """
    Compute token-level boundaries for each cell in a flattened table string.
    
    This function tokenizes a table where cells are separated by '||' delimiters
    and returns the start and end token indices for each cell. These boundaries
    are used for entity linking to identify which entities belong to which cells.
    
    Args:
        flattened_table: A string representation of a table with cells separated by '||' delimiters
    
    Returns:
        boundaries: A list of tuples (start_idx, end_idx) where each tuple represents
                   the token index range for one cell. 
    """
    tokens = preprocessor.tokenize(flattened_table)
    boundaries = []
    cell_start = None  # Start index of the current cell

    for i in range(len(tokens)):
        token = tokens[i]
        if token.text == "||" or token.text == "Ġ||":  # Cell delimiter
            if cell_start is not None:
                # Close the current cell
                boundaries.append((cell_start, i))
                cell_start = None
        else:
            if cell_start is None:  # Start a new cell
                cell_start = i

    return boundaries

def process_EL_res(col_spans):
    EL_col_res = []
    for span in col_spans:
        if len(span) < 1:
            EL_col_res.append("")
            continue
        item = span[0] # for each cell, only take the first entity linked
        item_string = item.__repr__()
        res = item_string.strip("[]").split(", ")
        qid_match = re.search(r'wikidata_entity_id=(\w+)', res[1])
        if qid_match:
            qid = qid_match.group(1)
            EL_col_res.append(qid)
        else:
            EL_col_res.append("")
            continue
    return EL_col_res


def safe_parse_dict(d):
    d = d.strip()
    if not d:
        return {}
    try:
        val = ast.literal_eval(d)
        return val if isinstance(val, dict) else {}
    except (SyntaxError, ValueError):
        return {}
    

def parse_pred(prediction): # parse the prediction returned by the row-linker-agent
    try:
        final_answer_start = prediction.find("FINAL ANSWER:") + len("FINAL ANSWER:")
        final_answer_str = prediction[final_answer_start:].strip()

        # Convert the string to a valid JSON (replace single quotes with double quotes)
        final_answer_json = json.loads(final_answer_str.replace("'", '"'))

        # Extract the type
        predicted_type = final_answer_json['type'][0]  # Get the first element in the list
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        predicted_type = "None"
        
    return predicted_type

def parse_json_pred(prediction):  # parse a JSON response
    try:
        # Extract JSON substring if there is a prefix like "FINAL ANSWER: "
        match = re.search(r'\{.*\}', prediction)
        if not match:
            return "NA"
        
        json_str = match.group(0)

        # Convert the string to a valid JSON (replace single quotes with double quotes)
        final_answer_json = json.loads(json_str.replace("'", '"'))

        # Extract the type
        predicted_type = final_answer_json['type'][0]  # Get the first element in the list
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        predicted_type = "None"
        

    return predicted_type

def f1_score_multilabel(true_list, pred_list):
    true_list = np.array(true_list)
    pred_list = np.array(pred_list)
    conf_mat = multilabel_confusion_matrix(np.array(true_list),
                                           np.array(pred_list))
    agg_conf_mat = conf_mat.sum(axis=0)
    # Note: Pos F1
    # [[TN FP], [FN, TP]] if we consider 1 as the positive class
    r = agg_conf_mat[1, 1] / agg_conf_mat[1, :].sum()
    p = agg_conf_mat[1, 1] / agg_conf_mat[:, 1].sum()

    TP = conf_mat[:, 1, 1].sum()  
    FP = conf_mat[:, 0, 1].sum()  
    FN = conf_mat[:, 1, 0].sum()  
    TN = conf_mat[:, 0, 0].sum()  
    # print("TP:", TP, "FP:", FP, "FN:", FN)
    
    micro_f1 = 2 * p * r / (p  + r) if (p + r) > 0 else 0.
    class_p = conf_mat[:, 1, 1] /  conf_mat[:, 1, :].sum(axis=1)
    class_r = conf_mat[:, 1, 1] /  conf_mat[:, :, 1].sum(axis=1)
    class_f1 = np.divide(2 * (class_p * class_r), class_p + class_r,
                         out=np.zeros_like(class_p), where=(class_p + class_r) != 0)
    class_f1 = np.nan_to_num(class_f1)
    macro_f1 = class_f1.mean()
    return (micro_f1, macro_f1, class_f1, conf_mat, p, r)

def parse_pred_RE(prediction): 
    try:
        final_answer_start = prediction.find("FINAL ANSWER:") + len("FINAL ANSWER:")
        final_answer_str = prediction[final_answer_start:].strip()

        # Convert the string to a valid JSON (replace single quotes with double quotes)
        final_answer_json = json.loads(final_answer_str.replace("'", '"'))

        # Extract the type
        predicted_type = final_answer_json['relation'][0]  # Get the first element in the list
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        predicted_type = "None"
        
    return predicted_type

def parse_json_pred_RE(prediction):  # parse a JSON response
    try:
        # Extract JSON substring if there is a prefix like "FINAL ANSWER: "
        match = re.search(r'\{.*\}', prediction)
        if not match:
            return "NA"
        
        json_str = match.group(0)

        # Convert the string to a valid JSON (replace single quotes with double quotes)
        final_answer_json = json.loads(json_str.replace("'", '"'))

        # Extract the type
        predicted_type = final_answer_json['relation'][0]  # Get the first element in the list
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        predicted_type = "None"

    return predicted_type

def extract_column_pair_rel(text):
    result = []
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if ':' in line:
            description = line.split(':', 1)[1].strip()
            result.append(description)
        else:
            result.append("")
            
    return result