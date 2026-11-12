"""refresh.py | Step 6 of the evidence milestone: bring the evidence up to date after new ratings, and say what changed.

    python main_backend.py rag refresh --check     # is the corpus older than its sources? (reads only)
    python main_backend.py rag refresh             # rebuild if a source changed, then report
    python main_backend.py rag refresh --force     # rebuild even if nothing changed
    python main_backend.py rag refresh --relabel   # also accept the labelling rules' new answer key

A refresh does, in order:

    1. check    the fingerprint (sha256) of every source (who_rated_what.csv, random-v2's rows.json and
                record.json) against the corpus manifest; nothing changed -> stop (unless --force)
    2. rebuild  the corpus from the sources (corpus.py), with a new manifest
    3. compare  old and new corpus by record ID: records added, removed, and changed (a new revision,
                that is a score given again, or a new text)
    4. labels   re-run each question's labelling rule on the new corpus and compare with labels.json:
                records the rule now adds or drops, and labelled records whose revision changed. Such a
                question is "stale": its answer key needs a look. labels.json is only rewritten with
                --relabel, so a person decides when the answer key changes
    5. rerun    the baseline on every question (history/rag/baseline_results.jsonl), so its results match
                the new corpus
    6. log      one line per refresh in history/rag/refresh_log.jsonl: when, why, what changed

A source that is missing is reported, and the refresh stops rather than build a corpus without it.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from backend import safe_files
from backend.genome import ratings
from backend.rag import baseline, corpus, labels

LOG_PATH = corpus.RAG_DIR / "refresh_log.jsonl"


def sources():
    out = [Path(ratings.who_path()), corpus.RANDOM_V2 / "rows.json", corpus.RANDOM_V2 / "record.json"]
    if (corpus.SANITY_ALL / "rows.json").is_file():          # every gene type alone, once it has been run
        out += [corpus.SANITY_ALL / "rows.json", corpus.SANITY_ALL / "record.json"]
    return out


def check(manifest_path=None, paths=None):
    """{"missing": [...], "changed": [...], "built": when the corpus was built} (no corpus yet: everything changed)."""
    manifest_path = Path(manifest_path or corpus.MANIFEST_PATH)
    paths = paths or sources()
    manifest = safe_files.read_json(manifest_path) if manifest_path.is_file() else {"sources": {}, "built": None}
    missing = [str(p) for p in paths if not Path(p).is_file()]
    changed = [str(p) for p in paths if Path(p).is_file() and manifest["sources"].get(str(p)) != corpus.sha256_of(p)]
    return {"missing": missing, "changed": changed, "built": manifest.get("built")}


def compare(old, new):
    """Records added, removed and changed (same ID, new revision or new text), by ID."""
    before, after = {r["id"]: r for r in old}, {r["id"]: r for r in new}
    changed = [i for i in sorted(before.keys() & after.keys())
               if (before[i]["revision"], before[i]["text"]) != (after[i]["revision"], after[i]["text"])]
    return {"added": sorted(after.keys() - before.keys()), "removed": sorted(before.keys() - after.keys()),
            "changed": changed}


def stale_labels(saved, records, questions):
    """Per question: what the rule would label now that is not in the answer key, and the reverse, and revisions moved."""
    now = labels.label(records, questions)
    revision = {r["id"]: r["revision"] for r in records}
    out = {}
    for qid, fresh in now.items():
        kept = saved["questions"].get(qid, {}).get("relevant", {})
        report = {"added": sorted(fresh["relevant"].keys() - kept.keys()),
                  "dropped": sorted(kept.keys() - fresh["relevant"].keys()),
                  "regraded": sorted(i for i in kept.keys() & fresh["relevant"].keys()
                                     if kept[i]["grade"] != fresh["relevant"][i]["grade"]),
                  "revised": sorted(i for i, x in kept.items() if i in revision and x["revision"] != revision[i])}
        if any(report.values()):
            out[qid] = report
    return out, now


def refresh(force=False, relabel=False, corpus_path=None, labels_path=None, log_path=LOG_PATH, rerun=True):
    corpus_path = Path(corpus_path or corpus.CORPUS_PATH)
    manifest_path = corpus_path.with_name(corpus_path.stem + "_manifest.json")
    state = check(manifest_path)
    if state["missing"]:
        raise SystemExit("Cannot refresh, a source is missing:\n  " + "\n  ".join(state["missing"]))
    if not state["changed"] and not force:
        return {"status": "up to date", "built": state["built"]}

    old = corpus.load(corpus_path) if corpus_path.is_file() else []
    new = corpus.build()
    manifest = corpus.write(new, corpus_path, sources=sources())
    diff = compare(old, new)

    questions = baseline.load_questions()
    labels_path = Path(labels_path or labels.LABELS_PATH)
    if labels_path.is_file():
        stale, fresh = stale_labels(labels.load(labels_path), new, questions)
    else:
        stale, fresh = {q["id"]: {"added": ["(no labels yet)"]} for q in questions}, labels.label(new, questions)
    if relabel:
        labels.write(fresh, manifest, labels_path)

    if rerun:
        baseline.run_all(baseline.BM25(new), questions, path=corpus_path.with_name("baseline_results.jsonl"))

    entry = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "why": "forced" if not state["changed"] else "sources changed",
             "sources_changed": state["changed"], "records": manifest["records"], "kinds": manifest["kinds"],
             "corpus_sha256": manifest["corpus_sha256"], "newest_revision": manifest["newest_revision"],
             "added": len(diff["added"]), "removed": len(diff["removed"]), "changed": len(diff["changed"]),
             "stale_questions": sorted(stale), "relabelled": relabel, "baseline_rerun": rerun}
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"status": "refreshed", **entry, "diff": diff, "stale": stale, "log": str(log_path)}


def show(result):
    if result["status"] == "up to date":
        print(f"Up to date: no source has changed since the corpus was built ({result['built']}).")
        return
    print(f"Refreshed ({result['why']}): {result['records']} records {result['kinds']}, newest revision {result['newest_revision']}")
    for path in result["sources_changed"]:
        print(f"  source changed: {path}")
    d = result["diff"]
    print(f"  records: {len(d['added'])} added, {len(d['removed'])} removed, {len(d['changed'])} changed")
    for name in ("added", "removed", "changed"):
        for i in d[name][:5]:
            print(f"    {name}: {i}")
        if len(d[name]) > 5:
            print(f"    ... and {len(d[name]) - 5} more {name}")
    if result["stale"]:
        print("  answer key: these questions need a look (rag label --show QID), or accept with --relabel:")
        for qid, r in result["stale"].items():
            print("    " + qid + ": " + ", ".join(f"{len(v)} {k}" for k, v in r.items() if v))
    else:
        print("  answer key: still matches every question")
    print(f"  baseline rerun: {'yes' if result['baseline_rerun'] else 'no'};  log: {result['log']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bring the evidence corpus up to date and report what changed.")
    parser.add_argument("--check", action="store_true", help="only say whether a source changed")
    parser.add_argument("--force", action="store_true", help="rebuild even if no source changed")
    parser.add_argument("--relabel", action="store_true", help="also rewrite labels.json from the rules")
    args = parser.parse_args(argv)
    if args.check:
        state = check()
        if state["missing"]:
            print("Missing sources:\n  " + "\n  ".join(state["missing"]))
        print("Up to date." if not state["changed"] else
              "Out of date, changed since " + str(state["built"]) + ":\n  " + "\n  ".join(state["changed"]))
        return
    show(refresh(force=args.force, relabel=args.relabel))


if __name__ == "__main__":
    main()
