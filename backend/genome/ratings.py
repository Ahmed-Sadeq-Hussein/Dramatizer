"""ratings.py | Listener ratings (MOS) of a genome search, the genomes ranked by them, and named genomes.

Up to 4 listeners rate each clip of a search (the plain baseline and every
genome) on a scale from 0 to 5. The mean opinion score (MOS) of a clip is the
mean of its listeners' scores. The genomes are then ranked by MOS, and the best
one can be picked and given a name, so it can be used again on other lines and
voices (a genome's genes hold positions in the sentence, not words).

    from backend.genome.ratings import save_scores, ranked, pick

    save_scores(run_dir, {"Ana": {"baseline": 3, "genome_01": 4.5}, "Ben": {"genome_01": 4}})
    ranked(run_dir)                          # the genomes, best MOS first
    pick(run_dir, 1, "Proud praise")         # name genome 1 and keep it in the library

Where it is stored (all under .gitignore: listener ratings are personal data):

    <run folder>/ratings.json        the run's current scores and picks, rewritten on every save
    history/ratings/mos.jsonl        every score ever saved, one line each, never rewritten,
                                     with the run, voice, line and markup, so ratings can be
                                     reviewed later even if a run folder is removed
    history/genomes/named.json       the named genomes: the genes, where they came from, their MOS

    python -m backend.genome.ratings               # every rated run, with its best genome
    python -m backend.genome.ratings --named       # the named genomes
    python -m backend.genome.ratings --export FILE # every score as CSV, for analysis
"""
import argparse
import csv
import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path

from backend import model
from backend.genome import search as genome_search

SCHEMA_VERSION = 1
MAX_RATERS = 4
LOWEST, HIGHEST = 0.0, 5.0
RATINGS_FILE = "ratings.json"
LOG_PATH = model.PROJECT_DIR / "history" / "ratings" / "mos.jsonl"
NAMED_PATH = model.PROJECT_DIR / "history" / "genomes" / "named.json"
CLIP = re.compile(r"^(baseline|genome_\d{2})$")
RATER = re.compile(r"^[\w .'-]{1,40}$")
NAME = re.compile(r"^[\w .,'!?()&+-]{1,60}$")

_lock = threading.Lock()   # the web page may save from two requests at once


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _write(path, data):
    """A temporary file first, so a crash never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def clip_key(number):
    """The key of a genome's clip: 1 -> "genome_01"."""
    return f"genome_{int(number):02d}"


def clean_rater(name):
    name = " ".join(str(name or "").split())
    if not RATER.match(name):
        raise ValueError(f"A listener's name must be 1 to 40 letters, digits, spaces or .'- : {name!r}")
    return name


def clean_name(name):
    name = " ".join(str(name or "").split())
    if not NAME.match(name):
        raise ValueError("A genome's name must be 1 to 60 characters: letters, digits, spaces and simple punctuation.")
    return name


def clean_score(value):
    score = float(value)
    if not LOWEST <= score <= HIGHEST:
        raise ValueError(f"A score must be from {LOWEST:g} to {HIGHEST:g}, not {score:g}.")
    return round(score, 2)


# --------------------------------------------------------------------------- #
# A run's ratings
# --------------------------------------------------------------------------- #

def load(run_dir):
    """The run's ratings: {"raters", "scores": {rater: {clip: score}}, "picked": [...]}; empty when none yet."""
    path = Path(run_dir) / RATINGS_FILE
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema_version": SCHEMA_VERSION, "run": Path(run_dir).name, "scale": [LOWEST, HIGHEST],
            "raters": [], "scores": {}, "picked": [], "updated": None}


def is_rated(run_dir):
    return any(load(run_dir)["scores"].values())


def save_scores(run_dir, scores, submitted_by=None):
    """
    Save the scores of one or more listeners: {rater: {clip: score}}, clip being
    "baseline" or "genome_NN". A score given again replaces the earlier one. Each
    is also appended to the review log. At most MAX_RATERS listeners per run.
    submitted_by: who sent them (the API key's owner), kept in the log.
    """
    run_dir = Path(run_dir)
    run = genome_search.load_run(run_dir)
    rendered = {"baseline"} if run["baseline"].get("audio") else set()
    rendered |= {clip_key(g["number"]) for g in run["genomes"] if g.get("audio")}

    cleaned = {}
    for rater, clips in scores.items():
        rater = clean_rater(rater)
        for clip, value in clips.items():
            if not CLIP.match(clip) or clip not in rendered:
                raise ValueError(f"No rendered clip {clip!r} in this run.")
            cleaned.setdefault(rater, {})[clip] = clean_score(value)
    if not cleaned:
        return load(run_dir)

    with _lock:
        ratings = load(run_dir)
        raters = list(ratings["raters"])
        for rater in cleaned:
            if rater not in raters:
                raters.append(rater)
        if len(raters) > MAX_RATERS:
            raise ValueError(f"At most {MAX_RATERS} listeners can rate one run; it already has "
                             f"{', '.join(ratings['raters'])}.")
        ratings["raters"] = raters
        time = _now()
        log = []
        genomes = {clip_key(g["number"]): g for g in run["genomes"]}
        for rater, clips in cleaned.items():
            ratings["scores"].setdefault(rater, {}).update(clips)
            for clip, score in clips.items():
                genome = genomes.get(clip)
                log.append({"time": time, "run": run["id"], "voice": run["voice"], "origin": run.get("origin"),
                            "text": run["text"], "rater": rater, "clip": clip, "score": score,
                            "markup": genome["markup"] if genome else run["text"],
                            "speed": genome["speed"] if genome else 1.0,
                            "genes": genome["genes"] if genome else [],
                            "verdict": genome["verdict"] if genome else "baseline",
                            "submitted_by": submitted_by})
        ratings["updated"] = time
        _write(run_dir / RATINGS_FILE, ratings)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            for entry in log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return ratings


def mos(ratings, clip):
    """{"mos", "n", "low", "high", "scores": {rater: score}} of one clip, or None when nobody rated it."""
    given = {rater: clips[clip] for rater, clips in ratings["scores"].items() if clip in clips}
    if not given:
        return None
    values = list(given.values())
    return {"mos": round(sum(values) / len(values), 2), "n": len(values), "low": min(values), "high": max(values),
            "scores": given}


def ranked(run_dir, run=None, ratings=None):
    """
    The run's genomes, best first: by MOS (genomes nobody rated come last), then
    fine before terrible, then the higher similarity in the sanity check. Each is
    the genome's dict plus "mos" (or None), "rank", "picked" (its name or None)
    and "above_baseline" (its MOS minus the baseline's, or None).
    """
    run = run or genome_search.load_run(run_dir)
    ratings = ratings or load(run_dir)
    picked = {p["genome"]: p["name"] for p in ratings["picked"]}
    baseline = mos(ratings, "baseline")
    items = []
    for genome in run["genomes"]:
        score = mos(ratings, clip_key(genome["number"]))
        items.append({**genome, "mos": score, "picked": picked.get(genome["number"]),
                      "above_baseline": round(score["mos"] - baseline["mos"], 2) if score and baseline else None})

    def order(item):
        similarity = (item.get("sanity") or {}).get("similarity") or 0
        return (item["mos"] is None, -(item["mos"]["mos"] if item["mos"] else 0),
                item.get("verdict") != "fine", -similarity, item["number"])

    items.sort(key=order)
    for rank, item in enumerate(items, start=1):
        item["rank"] = rank if item["mos"] else None
    return {"baseline": baseline, "genomes": items, "raters": ratings["raters"],
            "rated": sum(1 for item in items if item["mos"])}


# --------------------------------------------------------------------------- #
# Picking and naming a genome
# --------------------------------------------------------------------------- #

def load_named():
    return json.loads(NAMED_PATH.read_text(encoding="utf-8")) if NAMED_PATH.is_file() else {"genomes": []}


def pick(run_dir, number, name, named_by=None):
    """
    Pick genome `number` of a run and name it. The name is kept in the run's
    ratings and in the library of named genomes (unique names, ignoring case;
    naming the same genome again renames it).
    """
    run_dir = Path(run_dir)
    name = clean_name(name)
    run = genome_search.load_run(run_dir)
    genome = next((g for g in run["genomes"] if g["number"] == int(number)), None)
    if genome is None:
        raise ValueError(f"This run has no genome {number}.")

    with _lock:
        library = load_named()
        same_genome = lambda entry: entry["run"] == run["id"] and entry["genome"] == genome["number"]
        if any(e["name"].lower() == name.lower() and not same_genome(e) for e in library["genomes"]):
            raise ValueError(f"There is already a genome named {name!r}: choose another name.")

        ratings = load(run_dir)
        score = mos(ratings, clip_key(genome["number"]))
        entry = {
            "name": name, "named": _now(), "named_by": named_by, "run": run["id"], "genome": genome["number"],
            "genes": genome["genes"], "described": genome["described"], "markup": genome["markup"],
            "speed": genome["speed"], "text": run["text"], "voice": run["voice"], "origin": run.get("origin"),
            "verdict": genome["verdict"], "mos": score, "baseline_mos": mos(ratings, "baseline"),
        }
        library["genomes"] = [e for e in library["genomes"] if not same_genome(e)] + [entry]
        _write(NAMED_PATH, library)

        ratings["picked"] = [p for p in ratings["picked"] if p["genome"] != genome["number"]]
        ratings["picked"].append({"genome": genome["number"], "name": name, "time": entry["named"]})
        ratings["updated"] = entry["named"]
        _write(run_dir / RATINGS_FILE, ratings)
    return entry


def unpick(run_dir, number):
    """Take the name off a genome again (in the run and in the library)."""
    run_dir = Path(run_dir)
    run_id = genome_search.load_run(run_dir)["id"]
    with _lock:
        library = load_named()
        library["genomes"] = [e for e in library["genomes"] if not (e["run"] == run_id and e["genome"] == int(number))]
        _write(NAMED_PATH, library)
        ratings = load(run_dir)
        ratings["picked"] = [p for p in ratings["picked"] if p["genome"] != int(number)]
        _write(run_dir / RATINGS_FILE, ratings)


# --------------------------------------------------------------------------- #
# Reviewing later
# --------------------------------------------------------------------------- #

def rated_runs(runs_dir=None):
    """[(run folder, run dict, ranked(...))] for every run with at least one score, newest first."""
    out = []
    for folder, run in genome_search.list_runs(runs_dir):
        if (folder / RATINGS_FILE).is_file() and is_rated(folder):
            out.append((folder, run, ranked(folder, run)))
    return out


def read_log():
    if not LOG_PATH.is_file():
        return []
    return [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def export_csv(path):
    rows = read_log()
    fields = ["time", "run", "voice", "text", "rater", "clip", "score", "verdict", "speed", "markup"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review listener ratings (MOS) of genome searches.")
    parser.add_argument("--named", action="store_true", help="list the named genomes")
    parser.add_argument("--export", metavar="FILE", help="write every score ever saved as CSV")
    args = parser.parse_args()

    if args.export:
        print(f"{export_csv(args.export)} scores written to {args.export}")
    elif args.named:
        for entry in load_named()["genomes"]:
            score = f"MOS {entry['mos']['mos']:.2f} ({entry['mos']['n']} listeners)" if entry["mos"] else "not rated"
            print(f"{entry['name']}: genome {entry['genome']} of {entry['run']}, {score}\n    {entry['markup']}")
    else:
        for folder, run, rank in rated_runs():
            best = next((g for g in rank["genomes"] if g["mos"]), None)
            base = f"baseline MOS {rank['baseline']['mos']:.2f}" if rank["baseline"] else "baseline not rated"
            print(f"{run['id']}  ({', '.join(rank['raters'])}; {rank['rated']} genomes rated; {base})")
            if best:
                print(f"    best: genome {best['number']} MOS {best['mos']['mos']:.2f}  {best['markup']}")
