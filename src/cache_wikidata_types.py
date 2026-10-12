# -*- coding: utf-8 -*-
"""
Standalone, lightweight QID -> Wikidata entity-info caching loop.
Does NOT import utils.py / ReFinED / torch -- only needs SPARQLWrapper, so it
starts instantly and can be edited/restarted without reloading the 70GB ReFinED
model each time. Reads the unique QID list already collected in qid_list.json
(table/column context is unused here, only the QIDs matter) and writes results
into the SAME wikidata_cache.json file (under its "entity_full" section) that
utils.py's get_instance_of()/get_entity_info() read from, so the real pipeline
benefits for free. The query combines what used to be two separate live calls
(type lookup + label/description lookup) into one, halving request count.
"""
import json, time
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON

RACOON_SRC = Path(__file__).resolve().parent
QID_LIST_PATH = RACOON_SRC / "qid_list.json"  # written by the earlier ReFinED scan
CACHE_PATH = RACOON_SRC / "wikidata_cache.json"
SECTION = "entity_full"

def load_json(path, default):
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return default
    return default

def load_full_cache():
    data = load_json(CACHE_PATH, {})
    data.setdefault("entity_full", {})
    data.setdefault("relation", {})
    return data

def save_cache(full_cache):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(full_cache, f, ensure_ascii=False)

def query_entity_full(qid):
    """Same merged SPARQL query/parsing as utils.py's _fetch_entity_full(),
    duplicated here (not imported) specifically to avoid utils.py's
    module-level ReFinED load. One query gets both P31 "instance of" types
    AND the entity's own label/description."""
    sparql = SPARQLWrapper("http://query.wikidata.org/sparql")
    sparql.setQuery(f"""SELECT ?itemLabel ?itemDescription ?type ?typeLabel ?pq_obj ?pq_objLabel
       WHERE {{
       BIND(wd:{qid} AS ?item)
       OPTIONAL {{
           wd:{qid} p:P31 ?statementNode.
           ?statementNode ps:P31 ?type.
           OPTIONAL {{
              ?statementNode pq:P642 ?pq_obj.
           }}
       }}
       SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
       }}
       """)
    sparql.setReturnFormat(JSON)
    try:
        ret = sparql.query()
        data = ret.convert()["results"]["bindings"]
    except Exception:
        return None  # failure -- caller should NOT cache this

    if not data:
        return None

    label = data[0].get("itemLabel", {}).get("value", "")
    description = data[0].get("itemDescription", {}).get("value", "")

    qids, entity_set, raw_bindings = [], set(), []
    for binding in data:
        if "type" not in binding:
            continue
        entry = {
            "type_qid": binding["type"]["value"].split("/")[-1],
            "type_label": binding["typeLabel"]["value"],
        }
        if "pq_obj" in binding:
            entry["pq_obj_qid"] = binding["pq_obj"]["value"].split("/")[-1]
            entry["pq_obj_label"] = binding["pq_objLabel"]["value"]
        raw_bindings.append(entry)
        if "pq_obj" in binding:
            qids.append(entry["pq_obj_qid"]); entity_set.add(entry["pq_obj_label"])
        else:
            qids.append(entry["type_qid"]); entity_set.add(entry["type_label"])

    return {
        "label": label,
        "description": description,
        "qids": qids,
        "entity_set": list(entity_set),
        "raw_bindings": raw_bindings,
    }

def _run_loop():
    all_qids = load_json(QID_LIST_PATH, {})
    print(f"Loaded {len(all_qids)} distinct QIDs from {QID_LIST_PATH}", flush=True)

    attempt = 0
    while True:
        section = load_full_cache()[SECTION]
        pending = [q for q in all_qids if q not in section]
        if not pending:
            print("ALL QIDs CACHED. Done.", flush=True)
            break
        for qid in pending:
            attempt += 1
            full_cache = load_full_cache()  # re-read in case another process wrote to it
            if qid in full_cache[SECTION]:
                continue
            result = query_entity_full(qid)
            if result is not None and (result["entity_set"] or result["label"]):
                full_cache[SECTION][qid] = result
                save_cache(full_cache)
                n_cached = len(full_cache[SECTION])
                preview = result["label"] or (list(result["entity_set"])[:3] if result["entity_set"] else "")
                print(f"attempt {attempt}: qid={qid} -> SUCCESS ({preview}) | cached {n_cached}/{len(all_qids)}", flush=True)
            elif attempt % 20 == 0:
                print(f"attempt {attempt}: qid={qid} -> fail | cached {len(full_cache[SECTION])}/{len(all_qids)}", flush=True)
            time.sleep(65)

# RAG4CTA: guarded so this module can be imported (e.g. to reuse
# query_entity_full() for a one-off test) without triggering the long-running
# loop as a side effect -- that bug once accidentally started a second,
# colliding instance of this same loop.
if __name__ == "__main__":
    _run_loop()
