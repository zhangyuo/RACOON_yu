# -*- coding: utf-8 -*-
"""
Batch Wikidata lookups through the MediaWiki Action API (wbgetentities, up to 50 entities per request).

Replaces the one-QID-per-request SPARQL query on query.wikidata.org, which is rate limited to 1 request/minute
during a WDQS outage. fetch_entities() returns, per QID, a record in the format of wikidata_cache.json["entity_full"]:

  unchanged fields (what utils.py reads, same meaning as the SPARQL version):
    label, description, qids, entity_set, raw_bindings
  extra fields:
    aliases          English aliases
    subclass_of      [{"qid", "label"}]   P279 of the entity
    wikipedia_title  title of the English Wikipedia page ("" if none)
    statements       [{"pid", "property", "qid", "value"}] item-valued claims, coordinates as
                     {"pid", "property", "value": "lat, lon"}
    type_hierarchy   {type_qid: {"label", "instance_of": [{"qid","label"}], "subclass_of": [{"qid","label"}]}}
                     for every P31 type, the same two levels the Wikidata MCP hierarchy tool returns
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://www.wikidata.org/w/api.php"
UA = "RACOON-baseline/0.1 (af62lp@cs.aau.dk)"
BATCH = 50
MAX_STATEMENTS = 100

_labels = {}  # id -> English label, shared by all calls in this process


class NoSuchEntity(ValueError):
    """The API refused a batch because one id does not exist (deleted or never created)."""
    def __init__(self, entity_id, msg):
        super().__init__(msg)
        self.entity_id = entity_id


TRANSIENT_API_ERRORS = ("maxlag", "ratelimited", "internal_api_error_DBQueryError", "readonly")


def _get(params, retries=6):
    """GET with retries for transient failures only (HTTP 429/5xx, network errors, maxlag). Other API errors raise at once."""
    url = API + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
                d = json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                raise
            last = e
            wait = int(e.headers.get("Retry-After", 0) or 0) if e.code == 429 else 0
            time.sleep(min(wait or 2 ** attempt * 2, 120))
            continue
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            time.sleep(2 ** attempt * 2)
            continue
        if "error" in d:
            if d["error"].get("code") in TRANSIENT_API_ERRORS:
                last = d["error"]
                time.sleep(2 ** attempt * 2)
                continue
            if d["error"].get("code") == "no-such-entity" and d["error"].get("id"):
                raise NoSuchEntity(d["error"]["id"], str(d["error"])[:200])
            raise ValueError(f"Wikidata API error: {str(d['error'])[:200]}")
        return d
    raise RuntimeError(f"Wikidata API failed after {retries} tries: {last}")


def _entities(ids, props, **extra):
    """{requested id: entity} for the entities that exist (redirects are followed)."""
    out = {}
    ids = list(dict.fromkeys(ids))
    for i in range(0, len(ids), BATCH):
        batch = ids[i:i + BATCH]
        while batch:
            try:
                d = _get({"action": "wbgetentities", "ids": "|".join(batch), "props": props,
                          "languages": "en", "languagefallback": "1", "format": "json", **extra})
            except NoSuchEntity as e:  # one deleted id fails the whole request: drop it and retry the rest
                if e.entity_id not in batch:
                    raise
                batch.remove(e.entity_id)
                continue
            for k, e in d.get("entities", {}).items():
                if "missing" not in e:
                    out[k] = e
            break
    return out


def _ensure_labels(ids):
    todo = [i for i in dict.fromkeys(ids) if i and i not in _labels]
    for k, e in _entities(todo, "labels").items():
        _labels[k] = e.get("labels", {}).get("en", {}).get("value", k)
    for i in todo:
        _labels.setdefault(i, i)  # like the SPARQL label service: unlabeled -> the id itself


def _claims(e, pid):
    return e.get("claims", {}).get(pid, [])


def _qid(snak):
    dv = snak.get("datavalue")
    return dv["value"]["id"] if dv and dv.get("type") == "wikibase-entityid" else None


def _ref(qid):
    return {"qid": qid, "label": _labels[qid]}


def fetch_entities(qids):
    """{qid: record} for every QID that exists. Raises RuntimeError if the API keeps failing."""
    qids = list(dict.fromkeys(qids))
    ents = _entities(qids, "claims|labels|descriptions|aliases|sitelinks", sitefilter="enwiki")

    type_ids = {_qid(c["mainsnak"]) for e in ents.values() for c in _claims(e, "P31")} - {None}
    types = _entities(sorted(type_ids), "claims|labels") if type_ids else {}

    # every id whose English label is needed, fetched in as few requests as possible
    need = set(type_ids)
    for e in ents.values():
        for pid, cl in e.get("claims", {}).items():
            for c in cl[:MAX_STATEMENTS]:
                v = _qid(c["mainsnak"])
                if v:
                    need.update((pid, v))
                elif c["mainsnak"].get("datavalue", {}).get("type") == "globecoordinate":
                    need.add(pid)
                for q in c.get("qualifiers", {}).get("P642", []):
                    need.add(_qid(q))
    for t in types.values():
        for pid in ("P31", "P279"):
            need.update(_qid(c["mainsnak"]) for c in _claims(t, pid))
    _ensure_labels(need - {None})

    out = {}
    for qid, e in ents.items():
        rec = {"label": e.get("labels", {}).get("en", {}).get("value", qid),
               "description": e.get("descriptions", {}).get("en", {}).get("value", ""),
               "qids": [], "entity_set": [], "raw_bindings": []}
        entity_set = set()
        for c in _claims(e, "P31"):
            tq = _qid(c["mainsnak"])
            if not tq:
                continue
            quals = [q for q in (_qid(x) for x in c.get("qualifiers", {}).get("P642", [])) if q] or [None]
            for pq in quals:
                b = {"type_qid": tq, "type_label": _labels[tq]}
                if pq:
                    b["pq_obj_qid"], b["pq_obj_label"] = pq, _labels[pq]
                rec["raw_bindings"].append(b)
                rec["qids"].append(pq or tq)
                entity_set.add(_labels[pq] if pq else _labels[tq])
        rec["entity_set"] = list(entity_set)

        rec["aliases"] = [a["value"] for a in e.get("aliases", {}).get("en", [])]
        rec["subclass_of"] = [_ref(q) for q in (_qid(c["mainsnak"]) for c in _claims(e, "P279")) if q]
        rec["wikipedia_title"] = e.get("sitelinks", {}).get("enwiki", {}).get("title", "")

        stm = []
        for pid, cl in e.get("claims", {}).items():
            for c in cl:
                v = _qid(c["mainsnak"])
                dv = c["mainsnak"].get("datavalue")
                if v:
                    stm.append({"pid": pid, "property": _labels[pid], "qid": v, "value": _labels[v]})
                elif dv and dv.get("type") == "globecoordinate":
                    stm.append({"pid": pid, "property": _labels[pid],
                                "value": f"{dv['value']['latitude']:.5f}, {dv['value']['longitude']:.5f}"})
        rec["statements"] = stm[:MAX_STATEMENTS]

        rec["type_hierarchy"] = {}
        for tq in dict.fromkeys(b["type_qid"] for b in rec["raw_bindings"]):
            t = types.get(tq, {})
            rec["type_hierarchy"][tq] = {
                "label": _labels[tq],
                "instance_of": [_ref(q) for q in (_qid(c["mainsnak"]) for c in _claims(t, "P31")) if q],
                "subclass_of": [_ref(q) for q in (_qid(c["mainsnak"]) for c in _claims(t, "P279")) if q],
            }
        out[qid] = rec
    return out
