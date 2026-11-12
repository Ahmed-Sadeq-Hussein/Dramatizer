"""corpus.py | The evidence: one record per thing that can be cited, each with a source ID and a revision.

    python main_backend.py rag corpus          # build history/rag/corpus.jsonl from the records below
    python main_backend.py rag corpus --show clip:mos-diverse-v1_p1_CREMA-D_1014/genome_05

Three kinds of record, built from what the project already keeps:

    clip:<run>/<clip>         one listener's score for one clip of the listening batch (a genome or the
                              plain line): the voice, the listener, the score, the score against the same
                              listener's plain line, the genes in words, the sanity verdict, and whether an
                              artifact was heard. Revision: when that score was last given.
                              From history/ratings/who_rated_what.csv and the run's run.json.
    run:<run>                 one voice's listening run: the voice, the line, the pile, the listener, and
                              its genomes from best to worst. Revision: when its last score was given.
    sanity:random-v2/<id>     one genome of Milestone 2's random search and its sanity check: its genes,
                              what was heard, and why it failed (words, voice, pace). Revision: when the
                              search was made (record.json).
    sanity:sanity-all-genes/<id>   the same for one single-gene genome of the check of every gene type
                              on its own (backend/evaluation/sanity_all_genes.py).

Every record is {"id", "kind", "revision", "title", "text", "fields"}. "text" is what a retriever
searches; "fields" are the same facts as data, for answers and checks. The corpus is written to
history/rag/ (git-ignored: it holds listeners' scores), with a manifest of what it was built from.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

from backend import model, safe_files
from backend.genome import ratings, search

RAG_DIR = model.PROJECT_DIR / "history" / "rag"
CORPUS_PATH = RAG_DIR / "corpus.jsonl"
MANIFEST_PATH = RAG_DIR / "corpus_manifest.json"
RANDOM_V2 = model.PROJECT_DIR / "output" / "milestone2" / "random-v2"
SANITY_ALL = model.OUTPUT_DIR / "milestone2" / "sanity-all-genes"      # every gene type alone
BATCH = "mos-diverse-v1"


def gene_name(gene):
    """A gene's type in words: "pause (realization)", "vocalized noise"."""
    name = gene["feature"].replace("_", " ")
    return f"{name} ({gene['kind']})" if gene.get("kind") else name


def signed(x):
    return f"+{x:g}" if x > 0 else f"{x:g}"


def against_plain(above):
    if above is None:
        return None
    return "better than" if above > 0 else "equal to" if above == 0 else "worse than"


def _float(text):
    return None if text in (None, "") else float(text)


def clip_records(rows, runs):
    """One record per row of who_rated_what.csv (listening batch only)."""
    out = []
    for row in rows:
        run = runs.get(row["run"])
        if run is None or run.get("batch") != BATCH:
            continue
        number = int(row["genome"]) if row["genome"] else None
        genome = next((g for g in run["genomes"] if g["number"] == number), None) if number else None
        score, above = float(row["score"]), _float(row["above_plain_line"])
        artifact = {"yes": True, "no": False}.get(row.get("artifact") or "")
        voice, dataset = run["voice"], run["voice"].split("/")[0]
        if genome:
            types = sorted({gene_name(g) for g in genome["genes"]})
            title = f"{voice} genome {number}, scored {score:g} by {row['rater']}"
            text = (f"Voice {voice} ({dataset}), genome {number}, pile {row['pile']}. {row['rater']} scored it {score:g} "
                    f"out of 5, {signed(above)} against their score for the plain line, so {against_plain(above)} the "
                    f"plain line. It has {len(genome['genes'])} active features (genes): {'; '.join(genome['described'])}. "
                    f"Gene types: {', '.join(types)}. Speed {genome['speed']:g}. Sanity check: {genome['verdict']}. "
                    f"Artifact heard: {'not asked' if artifact is None else 'yes' if artifact else 'no'}. "
                    f"Markup: {genome['markup']}")
        else:
            types = []
            title = f"{voice} plain line, scored {score:g} by {row['rater']}"
            text = (f"Voice {voice} ({dataset}), the plain line with no genome (baseline), pile {row['pile']}. "
                    f"{row['rater']} scored it {score:g} out of 5. Line: {run['text']}")
        out.append({
            "id": f"clip:{row['run']}/{row['clip']}", "kind": "clip", "revision": row["rated_on"], "title": title, "text": text,
            "fields": {"voice": voice, "dataset": dataset, "run": row["run"], "clip": row["clip"], "genome": number,
                       "listener": row["rater"], "pile": int(row["pile"]), "score": score, "above_plain": above,
                       "features": len(genome["genes"]) if genome else 0, "gene_types": types,
                       "verdict": row["verdict"], "artifact": artifact, "changes": int(row["changes"] or 0),
                       "first_given": row["first_given"]}})
    return out


def run_records(clips, runs):
    """One record per listening run: its genomes from best to worst against the plain line."""
    by_run = {}
    for c in clips:
        by_run.setdefault(c["fields"]["run"], []).append(c)
    out = []
    for run_id, items in sorted(by_run.items()):
        run = runs[run_id]
        plain = next((c for c in items if c["fields"]["genome"] is None), None)
        genomes = sorted((c for c in items if c["fields"]["genome"] is not None),
                         key=lambda c: (-c["fields"]["score"], c["fields"]["genome"]))
        listener = items[0]["fields"]["listener"]
        order = ", ".join(f"genome {c['fields']['genome']} {c['fields']['score']:g} ({signed(c['fields']['above_plain'])})"
                          for c in genomes)
        best = genomes[0]["fields"]["score"] if genomes else None
        tops = [c["fields"]["genome"] for c in genomes if c["fields"]["score"] == best]
        better = sum(1 for c in genomes if c["fields"]["above_plain"] > 0)
        text = f"Listening run for voice {run['voice']}, pile {run.get('pile')}, scored by {listener}. Line: {run['text']} "
        if plain:
            text += f"Plain line scored {plain['fields']['score']:g}. "
        if genomes:
            text += (f"{better} of {len(genomes)} genomes scored better than the plain line. Best genome: "
                     f"{', '.join(str(t) for t in tops)} with {best:g}. All genomes from best to worst: {order}.")
        out.append({"id": f"run:{run_id}", "kind": "run", "revision": max(c["revision"] or "" for c in items),
                    "title": f"{run['voice']} listening run ({listener})", "text": text,
                    "fields": {"voice": run["voice"], "run": run_id, "listener": listener, "pile": run.get("pile"),
                               "best_genomes": tops, "better_than_plain": better, "genomes": len(genomes)}})
    return out


def sanity_records(folder=RANDOM_V2):
    """One record per judged genome of a sanity run (random-v2, or every gene type alone), with its check."""
    record = json.loads((folder / "record.json").read_text(encoding="utf-8"))
    rows = json.loads((folder / "rows.json").read_text(encoding="utf-8"))
    name = record.get("name") or folder.name
    out = []
    for row in rows:
        if row["verdict"] not in ("fine", "terrible"):        # did not render: nothing was checked
            continue
        key = f"{row['voice']}/line{row['line']}/genome{row['genome']:02d}"
        types = (sorted({gene_name(g) for g in row["genes"]}) if row.get("genes")
                 else sorted({f.replace("_", " ") for f in row["features"]}))
        reasons = [why for ok, why in ((row["wer_ok"], "words dropped or changed"),
                                       (row["similarity_ok"], "voice drifted from the speaker"),
                                       (row["pace_ok"], "pace broken (too slow, fast or long pauses)")) if not ok]
        verdict = "failed" if row["verdict"] == "terrible" else "passed"
        what = "Single-gene genome" if name == SANITY_ALL.name else "Random search genome"
        text = (f"{what} on voice {row['voice']} ({row['dataset']}), line {row['line']}, genome "
                f"{row['genome']}. Gene types: {', '.join(types)}. Sanity check {verdict}"
                + (f": {'; '.join(reasons)}." if reasons else ".")
                + f" Word error rate {row['wer']:.2f}, speaker similarity {row['similarity']:.2f}. "
                  f"Heard: {row.get('heard') or ''} Markup: {row['markup']}")
        out.append({"id": f"sanity:{name}/{key}", "kind": "sanity", "revision": record["made"],
                    "title": f"{name} {row['voice']} line {row['line']} genome {row['genome']}: sanity {verdict}",
                    "text": text,
                    "fields": {"voice": row["voice"], "dataset": row["dataset"], "line": row["line"], "genome": row["genome"],
                               "gene_types": types, "genes": len(row["features"]), "source": name,
                               "verdict": verdict, "reasons": reasons, "wer": row["wer"],
                               "similarity": row["similarity"]}})
    return out


def build(who_csv=None, runs_dir=None, random_v2=RANDOM_V2, sanity_all=SANITY_ALL):
    """Every record, in a fixed order (by kind, then ID), so the same inputs give the same corpus."""
    who_csv = Path(who_csv or ratings.who_path())
    with open(who_csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    runs = {run["id"]: run for _, run in search.list_runs(runs_dir)}
    clips = clip_records(rows, runs)
    records = clips + run_records(clips, runs)
    for folder in (random_v2, sanity_all):
        if folder and (folder / "rows.json").is_file():
            records += sanity_records(folder)
    order = {"clip": 0, "run": 1, "sanity": 2}
    records.sort(key=lambda r: (order[r["kind"]], r["id"]))
    return records


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else None


def write(records, path=CORPUS_PATH, sources=()):
    """The corpus as JSON lines, and a manifest: when, how many of each kind, and the hash of every input."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    temporary = safe_files.temporary_for(path)
    temporary.write_text(text, encoding="utf-8")
    safe_files.replace(temporary, path)
    kinds = {}
    for r in records:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    manifest = {"built": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "records": len(records), "kinds": kinds,
                "corpus_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "newest_revision": max((r["revision"] or "" for r in records), default=None),
                "sources": {str(s): sha256_of(s) for s in sources}}
    safe_files.write_json(path.with_name(path.stem + "_manifest.json"), manifest)
    return manifest


def load(path=CORPUS_PATH):
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"No corpus yet at {path}: run  python main_backend.py rag corpus")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the evidence corpus (history/rag/corpus.jsonl).")
    parser.add_argument("--show", metavar="ID", help="print one record of the corpus already built")
    args = parser.parse_args(argv)
    if args.show:
        found = next((r for r in load() if r["id"] == args.show), None)
        print(json.dumps(found, indent=2, ensure_ascii=False) if found else f"No record {args.show!r}.")
        return
    records = build()
    from backend.rag.refresh import sources
    manifest = write(records, sources=sources())
    print(f"{manifest['records']} records {manifest['kinds']} -> {CORPUS_PATH}")


if __name__ == "__main__":
    main()
