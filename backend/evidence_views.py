"""evidence_views.py | The evidence routes' answers, made from the evidence files alone (no web server, no login).

The API (backend/api/server.py) and the hosted copy of the results (API calls/remote_results.py, reading
what publish_results.py published) both answer with these functions, so the two give the same answers
to the same question. Who may see what is decided by the caller: `listener` is None for someone who may
see every listener's scores, else the one listener whose scores are shown.

A request that cannot be answered raises Refused(message, status), as the API would.
"""
from collections import Counter
from urllib.parse import urlencode

GROUPS = ("listener", "voice", "dataset", "gene_type", "genes", "verdict", "genome", "pile")
SORTS = ("score", "above_plain", "rated_on", "mean", "mean_above_plain", "count", "beat_plain")


class Refused(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


def may_read(record, listener=None):
    """Rankings and other listeners' scores are only for those who may see everyone's (listener None)."""
    if listener is None:
        return True
    return not (record["kind"] == "run" or record["kind"] == "clip" and record["fields"]["listener"] != listener)


def provenance(records, snapshot=None, listener=None):
    """{snapshot_id, ratings_up_to, data_revision, scope}: the data state an answer was made from."""
    revision = max((r["revision"] for r in records if r["revision"] and may_read(r, listener)), default=None)
    if listener is None and snapshot is not None:
        return {"snapshot_id": snapshot["provenance"]["snapshot_id"],
                "ratings_up_to": snapshot["provenance"]["ratings_up_to"], "data_revision": revision,
                "scope": "every listener"}
    own = [r["revision"] for r in records if r["kind"] == "clip" and r["fields"]["listener"] == listener]
    return {"snapshot_id": None, "ratings_up_to": max(own, default=None), "data_revision": revision,
            "scope": "your own scores"}


def score_rows(records, genomes, listener=None):
    """Every score the caller may see, one row per listener per clip. genomes: {run: {number: (described, markup)}}."""
    rows = []
    for r in records:
        if r["kind"] != "clip":
            continue
        f = r["fields"]
        if listener is not None and f["listener"] != listener:
            continue
        run = genomes.get(f["run"], {})
        described, markup = (run.get(f["genome"]) or run.get(str(f["genome"])) or ([], None)) if f["genome"] else ([], None)
        rows.append({"listener": f["listener"], "voice": f["voice"], "dataset": f["dataset"], "pile": f["pile"],
                     "search": f["run"], "clip": f["clip"], "genome": f["genome"], "score": f["score"],
                     "above_plain": f["above_plain"], "genes": f["features"], "gene_types": f["gene_types"],
                     "genes_described": described, "spoken_as": markup, "verdict": f["verdict"],
                     "artifact_heard": f["artifact"], "rated_on": r["revision"]})
    return rows


def matches(row, args):
    def has(name, value):
        return value.lower() in str(row[name]).lower()
    kind = args.get("clip", "genome")
    if kind == "genome" and row["genome"] is None or kind == "plain" and row["genome"] is not None:
        return False
    for name in ("listener", "voice", "dataset", "verdict"):
        if args.get(name) and not has(name, args[name]):
            return False
    if args.get("gene_type") and not any(args["gene_type"].lower() == t.lower() for t in row["gene_types"]):
        return False
    if args.get("genes") not in (None, "") and str(row["genes"]) != str(args["genes"]):
        return False
    if args.get("genome") not in (None, "") and str(row["genome"]) != str(args["genome"]):
        return False
    if args.get("pile") not in (None, "") and str(row["pile"]) != str(args["pile"]):
        return False
    if args.get("search") and args["search"] != row["search"]:
        return False
    if args.get("from") and row["rated_on"] < args["from"] or args.get("to") and row["rated_on"] > args["to"] + "~":
        return False
    return True


def group(rows, by):
    groups = {}
    for row in rows:
        keys = row["gene_types"] if by == "gene_type" else [f"{row['voice']} genome {row['genome']}" if by == "genome"
                                                           else row[by]]
        for k in keys or ["(none)"]:
            groups.setdefault(str(k), []).append(row)
    out = []
    for k, members in groups.items():
        scores = [m["score"] for m in members]
        above = [m["above_plain"] for m in members if m["above_plain"] is not None]
        out.append({by: k, "count": len(members), "mean": round(sum(scores) / len(scores), 3),
                    "low": min(scores), "high": max(scores),
                    "mean_above_plain": round(sum(above) / len(above), 3) if above else None,
                    "beat_plain": round(sum(a > 0 for a in above) / len(above), 3) if above else None,
                    "listeners": sorted({m["listener"] for m in members})})
    return out


def scores(all_rows, args):
    """GET /evidence/scores without the note: the rows (or groups) matching args, sorted and limited."""
    args = {k: str(v).strip()[:80] for k, v in args.items() if v is not None}
    by = args.get("group_by") or None
    if by and by not in GROUPS:
        raise Refused(f"group_by is one of: {', '.join(GROUPS)}.")
    sort = args.get("sort") or ("mean" if by else "score")
    if sort not in SORTS:
        raise Refused(f"sort is one of: {', '.join(SORTS)}.")
    try:
        limit = max(1, min(200, int(args.get("limit") or 20)))
    except ValueError:
        raise Refused("limit is a whole number.")
    rows = [r for r in all_rows if matches(r, args)]
    items = group(rows, by) if by else rows
    if by and sort in ("score", "above_plain", "rated_on"):
        sort = "mean" if sort == "score" else "mean_above_plain" if sort == "above_plain" else "count"
    if not by and sort not in ("score", "above_plain", "rated_on"):
        sort = "score"
    present = [i for i in items if i.get(sort) is not None]
    present.sort(key=lambda i: i[sort], reverse=args.get("order", "desc") != "asc")
    shown = present[:limit] + ([] if len(present) >= limit else [i for i in items if i.get(sort) is None][:limit - len(present)])
    filters = {k: v for k, v in args.items() if k not in ("limit", "order", "sort", "group_by") and v}
    return {"filters": filters, "group_by": by, "sort": sort, "total": len(items), "rows_matched": len(rows),
            "listeners": sorted({r["listener"] for r in rows}), ("groups" if by else "rows"): shown}


def search(records, index, q, kind=None, k=6, listener=None):
    """GET /evidence/search without the provenance: the records best matching q (BM25)."""
    from backend.rag import baseline

    q = " ".join(str(q or "").split())[:300]
    if not q:
        raise Refused("Ask something: ?q=words")
    kind = kind or None
    if kind not in (None, "clip", "run", "sanity"):
        raise Refused("kind is clip, run or sanity.")
    try:
        k = max(1, min(20, int(k or 6)))
    except ValueError:
        raise Refused("k is a whole number.")
    query = list(dict.fromkeys(baseline.words(q)))
    scored = []
    for i, r in enumerate(records):                    # by position: a clip's ID is shared by its listeners
        if kind not in (None, r["kind"]) or not may_read(r, listener):
            continue
        score = index.score(query, i)
        if score > 0:
            scored.append((score, i))
    scored.sort(key=lambda x: (-x[0], records[x[1]]["id"], x[1]))
    return {"question": q, "hits": [{"rank": n, "id": records[i]["id"], "kind": records[i]["kind"],
                                     "revision": records[i]["revision"], "title": records[i]["title"],
                                     "text": records[i]["text"], "score": round(sc, 3)}
                                    for n, (sc, i) in enumerate(scored[:k], start=1)]}


def answer(records, index, q, listener=None):
    """GET /evidence/answer without the provenance: a cited answer, or "nothing answers this"."""
    from backend.rag import answer as rag_answer

    q = " ".join(str(q or "").split())[:300]
    if not q:
        raise Refused("Ask something: ?q=words")
    a = rag_answer.answer(q, records, index, listener=listener)
    a.pop("cited_ids", None)
    return a


def question(number, charts, snapshot, chart_url=None):
    """GET /evidence/questions/<n>: the chart's claim and note, with the snapshot's provenance.
    charts: what evidence_charts.make_charts returned. chart_url: where the chart's JPEG is, if anywhere."""
    found = [c for c in charts["charts"] if str(c["number"]) == str(number)]
    if not found:
        raise Refused(f"There are {len(charts['charts'])} questions, 1 to {len(charts['charts'])}.", 404)
    chart = found[0]
    out = {"number": int(number), "question": chart["question"], "range": charts["range"],
           "genome_scores": charts["genome_scores"], "sanity_checks": charts["sanity_checks"],
           "sanity_source": charts.get("sanity_source"),
           "provenance": {k: snapshot["provenance"][k] for k in ("snapshot_id", "ratings_up_to", "code_commit", "rule")},
           "listeners_per_genome": snapshot["coverage"]["listeners_per_genome"], "caveats": snapshot["caveats"]}
    if int(number) == 1:                              # a winner only where the snapshot's rule allows one
        out["voice_status"] = dict(Counter(v["status"] for v in snapshot["voices"]))
        out["winner_rule"] = snapshot["provenance"]["winner_rule"]
    if "skipped" in chart:
        return {**out, "skipped": chart["skipped"]}
    return {**out, "answer": chart["claim"], "keep_in_mind": chart["note"], "chart": chart_url}


def range_query(start, end):
    return "?" + urlencode({k: v for k, v in (("from", start), ("to", end)) if v}) if start or end else ""
