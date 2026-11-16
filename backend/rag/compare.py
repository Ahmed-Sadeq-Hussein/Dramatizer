"""compare.py | Step 5 of the evidence milestone: the baseline against the variant, scored with the answer key.

    python main_backend.py rag compare                        # bm25, dense and hybrid on every question
    python main_backend.py rag compare --show 5               # and each question's top 5 per retriever

Every retriever answers the questions in questions.json on the same corpus, and its top 10
records are scored against labels.json (step 2), which was made by written rules from the
records' fields, never from a retriever's output:

    hit@k      1 if a labelled record (grade 1 or 2) is in the top k, else 0       (k = 1, 3, 5, 10)
    MRR        1 / the rank of the first labelled record in the top 10, 0 if none
    P@10       the share of the top 10 that is labelled: for summary questions, whose answer
               needs many records, this says more than hit@k
    nDCG@10    the ranking weighted by grade (2 counts more than 1) and position, 1 = ideal

The means are over the supported questions only. An unsupported question (q09, q10) has no
right record, so it cannot be scored this way; its top score is kept, which is what a
safe-absence cut-off would use. With 8 scored questions, a difference is a direction, not a
proof: the report also counts on how many questions the variant beat, tied or lost to the
baseline.

The comparison is refused if labels.json was made from another corpus than the one searched
(run  python main_backend.py rag refresh --relabel  first). Results, with each question's top
records for inspection, go to history/rag/compare_results.json (git-ignored: the record
titles hold listeners' names and scores).
"""
import argparse
import math
from datetime import datetime

from backend import safe_files
from backend.rag import baseline, dense, labels
from backend.rag import corpus as evidence

RETRIEVERS = ["bm25", "dense", "hybrid"]
KS = [1, 3, 5, 10]
DEPTH = 10
RESULTS_PATH = evidence.RAG_DIR / "compare_results.json"


def scores(hits, relevant):
    """The measures for one ranked list; relevant = {record id: grade}."""
    ids = [h["id"] for h in hits[:DEPTH]]
    first = next((rank for rank, i in enumerate(ids, start=1) if i in relevant), None)
    out = {f"hit@{k}": int(any(i in relevant for i in ids[:k])) for k in KS}
    out["mrr"] = 1.0 / first if first else 0.0
    out["p@10"] = sum(i in relevant for i in ids) / DEPTH
    dcg = sum((2 ** relevant.get(i, 0) - 1) / math.log2(rank + 1) for rank, i in enumerate(ids, start=1))
    ideal = sorted(relevant.values(), reverse=True)[:DEPTH]
    idcg = sum((2 ** g - 1) / math.log2(rank + 1) for rank, g in enumerate(ideal, start=1))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    out["first_relevant_rank"] = first
    return out


def mean(rows, key):
    return round(sum(r[key] for r in rows) / len(rows), 3) if rows else None


def compare(records, questions, answer_key, retrievers, depth=DEPTH):
    """{"questions": [...per question...], "means": {retriever: {group: {measure: mean}}}, "paired": {...}}"""
    per_question = []
    for q in questions:
        key = answer_key[q["id"]]
        relevant = {i: x["grade"] for i, x in key["relevant"].items()}
        row = {"id": q["id"], "question": q["question"], "kind": q["kind"], "supported": q["supported"],
               "expect": key["expect"], "relevant": len(relevant), "by": {}}
        for name, retriever in retrievers.items():
            hits = retriever.search(q["question"], k=depth)
            for h in hits:
                h["grade"] = relevant.get(h["id"], 0)
            row["by"][name] = {"top_score": hits[0]["score"] if hits else 0.0, "hits": hits,
                               **(scores(hits, relevant) if relevant else {})}
        per_question.append(row)

    measures = [f"hit@{k}" for k in KS] + ["mrr", "p@10", "ndcg@10"]
    scored = [r for r in per_question if r["relevant"]]
    groups = {"all": scored, "lookup": [r for r in scored if r["kind"] == "lookup"],
              "summary": [r for r in scored if r["kind"] == "summary"]}
    means = {name: {g: {m: mean([r["by"][name] for r in rows], m) for m in measures} | {"questions": len(rows)}
                    for g, rows in groups.items()} for name in retrievers}
    paired = {}
    if "bm25" in retrievers:
        for name in retrievers:
            if name == "bm25":
                continue
            d = [r["by"][name]["mrr"] - r["by"]["bm25"]["mrr"] for r in scored]
            paired[name] = {"measure": "mrr", "better": sum(x > 1e-9 for x in d), "same": sum(abs(x) <= 1e-9 for x in d),
                            "worse": sum(x < -1e-9 for x in d)}
    return {"questions": per_question, "means": means, "paired_with_bm25": paired}


def check_labels(answer_key, manifest):
    if answer_key.get("corpus_sha256") != manifest.get("corpus_sha256"):
        raise SystemExit("labels.json was made from another corpus than history/rag/corpus.jsonl.\n"
                         "Run  python main_backend.py rag refresh --relabel  (and look at what changed) first.")


def table(result):
    measures = ["hit@1", "hit@3", "hit@5", "hit@10", "mrr", "p@10", "ndcg@10"]
    lines = []
    for group in ("all", "lookup", "summary"):
        n = next(iter(result["means"].values()))[group]["questions"]
        lines.append(f"\n{group} ({n} questions)\n  {'':8s}" + "".join(f"{m:>9s}" for m in measures))
        for name, by in result["means"].items():
            lines.append(f"  {name:8s}" + "".join(f"{by[group][m]:9.3f}" if by[group][m] is not None else f"{'-':>9s}"
                                                  for m in measures))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare the retrievers on the labelled questions (hit@k, MRR).")
    parser.add_argument("--retrievers", default=",".join(RETRIEVERS), help="comma-separated: bm25, dense, hybrid")
    parser.add_argument("--show", type=int, default=0, metavar="N", help="print each question's top N per retriever")
    args = parser.parse_args(argv)

    records = evidence.load()
    answer_key = labels.load()
    check_labels(answer_key, safe_files.read_json(evidence.MANIFEST_PATH))
    names = [n.strip() for n in args.retrievers.split(",") if n.strip()]
    shared_dense = dense.Dense(records) if {"dense", "hybrid"} & set(names) else None
    retrievers = {}
    for n in names:
        if n == "bm25":
            retrievers[n] = baseline.BM25(records)
        elif n == "dense":
            retrievers[n] = shared_dense
        elif n == "hybrid":
            retrievers[n] = dense.Hybrid(records, dense=shared_dense)
        else:
            raise SystemExit(f"No retriever {n!r}: {', '.join(RETRIEVERS)}")
    result = compare(records, baseline.load_questions(), answer_key["questions"], retrievers)
    result.update({"made": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "corpus_sha256": answer_key["corpus_sha256"],
                   "records": len(records), "settings": {"bm25": {"k1": baseline.K1, "b": baseline.B},
                                                         "dense": {"model": dense.MODEL_NAME, "max_tokens": dense.MAX_TOKENS},
                                                         "hybrid": {"rrf_k": dense.RRF_K, "pool": dense.POOL}}})
    safe_files.write_json(RESULTS_PATH, result)

    for r in result["questions"]:
        line = f"\n{r['id']} [{r['kind']}, {r['relevant']} labelled] {r['question']}\n"
        for name, x in r["by"].items():
            line += (f"  {name:7s} MRR {x['mrr']:.3f}  hit@10 {x['hit@10']}  P@10 {x['p@10']:.1f}  nDCG {x['ndcg@10']:.3f}"
                     if r["relevant"] else f"  {name:7s} (nothing should be found)  top score {x['top_score']:.4f}") + "\n"
            for h in x["hits"][:args.show]:
                line += f"      {h['rank']:2d}. [{h['grade'] or '-'}] {h['id']}  {h['title']}\n"
        print(line.rstrip())
    print(table(result))
    for name, p in result["paired_with_bm25"].items():
        print(f"\n{name} against bm25, MRR per question: better on {p['better']}, the same on {p['same']}, worse on {p['worse']}")
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
