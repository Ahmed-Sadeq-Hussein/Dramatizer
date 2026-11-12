"""labels.py | Step 2 of the evidence milestone: which records answer each question (the answer key).

    python main_backend.py rag label            # label every question from the corpus -> backend/rag/labels.json
    python main_backend.py rag label --show q07 # the labelled records of one question

Each question has a written rule that picks its evidence from the records' fields
(scores, genes, verdicts), never from what a retriever returns, so the labels can
judge any retriever fairly. The rules are below in RULES, one per question, each
with the sentence that says what it means. A record is labelled with a grade:

    2   the record answers the question by itself (or a part of it: one voice of "each voice")
    1   the record is needed with others to answer it, or supports a grade-2 record

Questions the evidence cannot answer (q09, q10) have no relevant records: the
right result is to say that nothing answers them (safe absence, a later step).

Every labelled record keeps its revision (when its score was given), and labels.json
keeps the corpus fingerprint, so a refresh can tell when a label may be out of date
(refresh.py). The labels file holds record IDs only, no scores, so it can be committed.
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from backend import safe_files
from backend.rag import corpus as evidence

LABELS_PATH = Path(__file__).resolve().parent / "labels.json"
LOWER_CHANCE = {"hissing", "speed", "vocalized noise"}     # the result of question 6 (evidence chart q6)


def _genome_clips(records):
    return [r for r in records if r["kind"] == "clip" and r["fields"]["genome"] is not None]


def _by_kind(records, kind):
    return [r for r in records if r["kind"] == kind]


def best_per_voice(records):
    """q01: each voice's run record (2: it names the best genome), and the clips of its best genomes (1)."""
    out = {r["id"]: 2 for r in _by_kind(records, "run")}
    best = defaultdict(float)
    clips = _genome_clips(records)
    for c in clips:
        best[c["fields"]["run"]] = max(best[c["fields"]["run"]], c["fields"]["score"])
    for c in clips:
        if c["fields"]["score"] == best[c["fields"]["run"]]:
            out[c["id"]] = 1
    return out


def artifacts(records):
    """q02: clips where a listener heard an artifact (2), and genomes that failed the sanity check (1),
    the closest evidence while no listener has answered the artifact question."""
    out = {r["id"]: 2 for r in _genome_clips(records) if r["fields"]["artifact"] is True}
    out.update({r["id"]: 1 for r in _by_kind(records, "sanity") if r["fields"]["verdict"] == "failed"})
    return out


def bad_mos(records):
    """q03: genomes rated worse than the plain line of the same voice by the same listener (2)."""
    return {r["id"]: 2 for r in _genome_clips(records) if r["fields"]["above_plain"] < 0}


def better_than_plain(records):
    """q04: genomes rated better than their plain line (2), and the runs that have at least one (1)."""
    out = {r["id"]: 2 for r in _genome_clips(records) if r["fields"]["above_plain"] > 0}
    out.update({r["id"]: 1 for r in _by_kind(records, "run") if r["fields"]["better_than_plain"] > 0})
    return out


def by_feature_count(records):
    """q05: every rated genome (1): the answer counts all of them by their number of features."""
    return {r["id"]: 1 for r in _genome_clips(records)}


def exclude(records):
    """q06: rated genomes with a gene type given a lower chance (hissing, speed, vocalized noise) that did
    worse than their plain line (2), the same gene types' other rated genomes (1), and their failed sanity checks (1)."""
    out = {}
    for r in _genome_clips(records):
        if LOWER_CHANCE & set(r["fields"]["gene_types"]):
            out[r["id"]] = 2 if r["fields"]["above_plain"] < 0 else 1
    for r in _by_kind(records, "sanity"):
        if r["fields"]["verdict"] == "failed" and LOWER_CHANCE & set(r["fields"]["gene_types"]):
            out[r["id"]] = 1
    return out


def one_genome(voice, genome):
    def rule(records):
        out = {}
        for r in records:
            f = r["fields"]
            if r["kind"] == "clip" and f["voice"] == voice and f["genome"] == genome:
                out[r["id"]] = 2                     # the genome, its genes and its score
            elif r["kind"] == "clip" and f["voice"] == voice and f["genome"] is None:
                out[r["id"]] = 1                     # the plain line it is compared with
            elif r["kind"] == "run" and f["voice"] == voice:
                out[r["id"]] = 1                     # its place among the voice's genomes
        return out
    rule.__doc__ = f"q07: the clip of genome {genome} of {voice} (2), its plain line and its run (1)."
    return rule


def voice_better(voice):
    def rule(records):
        out = {}
        for r in records:
            f = r["fields"]
            if r["kind"] == "run" and f["voice"] == voice:
                out[r["id"]] = 2                     # lists every genome against the plain line
            elif r["kind"] == "clip" and f["voice"] == voice and f["genome"] is not None and f["above_plain"] > 0:
                out[r["id"]] = 2
            elif r["kind"] == "clip" and f["voice"] == voice and f["genome"] is None:
                out[r["id"]] = 1
        return out
    rule.__doc__ = f"q08: the run of {voice} and its genomes better than the plain line (2), the plain line (1)."
    return rule


def nothing(records):
    """No record answers this question: the right result is to say so."""
    return {}


RULES = {
    "q01": best_per_voice, "q02": artifacts, "q03": bad_mos, "q04": better_than_plain, "q05": by_feature_count,
    "q06": exclude, "q07": one_genome("CREMA-D/1014", 5), "q08": voice_better("VCTK/p250"),
    "q09": nothing, "q10": nothing,
}


def label(records, questions):
    """{question id: {"rule", "expect", "relevant": {record id: {"grade", "revision"}}}}"""
    revision = {r["id"]: r["revision"] for r in records}
    out = {}
    for q in questions:
        rule = RULES[q["id"]]
        found = rule(records)
        out[q["id"]] = {"question": q["question"], "rule": " ".join(rule.__doc__.split()),
                        "expect": "answer" if found else "decline",
                        "relevant": {i: {"grade": g, "revision": revision[i]} for i, g in sorted(found.items())}}
    return out


def write(labels, manifest, path=LABELS_PATH):
    data = {"about": "Step 2 of the evidence milestone: the records that answer each question (grade 2: answers it, "
                     "1: needed with others). Made by the rules in backend/rag/labels.py; check with "
                     "python main_backend.py rag refresh.",
            "made": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "corpus_sha256": manifest["corpus_sha256"],
            "records": manifest["records"], "questions": labels}
    safe_files.write_json(path, data)
    return data


def load(path=LABELS_PATH):
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"No labels yet at {path}: run  python main_backend.py rag label")
    return safe_files.read_json(path)


def main(argv=None):
    from backend.rag import baseline

    parser = argparse.ArgumentParser(description="Label which records answer each question (backend/rag/labels.json).")
    parser.add_argument("--show", metavar="QUESTION", help="print the labelled records of one question")
    args = parser.parse_args(argv)
    if args.show:
        q = load()["questions"].get(args.show)
        if q is None:
            raise SystemExit(f"No question {args.show!r}.")
        print(f"{args.show}: {q['question']}\nRule: {q['rule']}\nExpect: {q['expect']}, {len(q['relevant'])} records")
        for i, x in q["relevant"].items():
            print(f"  {x['grade']}  {i}  (revision {x['revision']})")
        return
    records = evidence.load()
    labels = label(records, baseline.load_questions())
    write(labels, safe_files.read_json(evidence.MANIFEST_PATH))
    for qid, q in labels.items():
        grades = [x["grade"] for x in q["relevant"].values()]
        print(f"{qid}  {q['expect']:7s}  {grades.count(2):4d} answer  {grades.count(1):4d} support   {q['question']}")
    print(f"\nSaved: {LABELS_PATH}")


if __name__ == "__main__":
    main()
