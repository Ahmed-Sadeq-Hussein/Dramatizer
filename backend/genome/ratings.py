"""ratings.py | Listener ratings (MOS) of a genome search, the genomes ranked by them, and named genomes.

Up to 4 listeners rate each clip of a search (the plain baseline and every
genome) on a scale from 0 to 5. The mean opinion score (MOS) of a clip is the
mean of its listeners' scores. The genomes are then ranked by MOS, and the best
one can be picked and given a name, so it can be used again on other lines and
voices (a genome's genes hold positions in the sentence, not words).

With each score a listener also says whether they heard an artifact in the clip
(a glitch, a click, noise, a cut-off or a garbled word): yes or no. Scores given
before this was asked have no answer (None).

    from backend.genome.ratings import save_scores, ranked, pick

    save_scores(run_dir, {"Ana": {"baseline": 3, "genome_01": 4.5}, "Ben": {"genome_01": 4}},
                artifacts={"Ana": {"genome_01": True, "baseline": False}})
    ranked(run_dir)                          # the genomes, best MOS first
    pick(run_dir, 1, "Proud praise")         # name genome 1 and keep it in the library

Where it is stored (all under .gitignore: listener ratings are personal data):

    the project's database           (backend/storage.py, history/dramatizer.db) the truth:
        rating_runs, ratings         each run's raters and picks, and every listener's current score
                                     of every clip: the score, the artifact answer, when it was first
                                     given and last changed, and the account that gave it
        rating_log                   every score ever given or changed, never updated or deleted (the
                                     database refuses it): when, who (name and login account), the run,
                                     batch, pile, voice, line, clip, markup, the artifact answer, and
                                     the score and answer before
    <run folder>/ratings.json        a copy of the run's ratings, written after every change, for
                                     reading the folder on its own (exports, import_mos.py). A run
                                     folder that has a ratings.json but is not in the database yet
                                     (copied in from elsewhere) is read from it, and saved to the
                                     database on its first change.
    history/ratings/who_rated_what.csv
                                     the current score of every listener for every clip, with its
                                     date: rebuilt after every save, open it in a spreadsheet
    history/genomes/named.json       the named genomes: the genes, where they came from, their MOS

    python -m backend.genome.ratings               # every rated run, with its best genome
    python -m backend.genome.ratings --named       # the named genomes
    python -m backend.genome.ratings --who         # who has scored what, per listener and pile
    python -m backend.genome.ratings --export FILE # every score ever given (the log) as CSV
    python -m backend.genome.ratings --current FILE  # the current scores (who rated what) as CSV
"""
import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

from backend import model, safe_files, storage
from backend.genome import search as genome_search

SCHEMA_VERSION = 1
MAX_RATERS = 4
LOWEST, HIGHEST = 0.0, 5.0
RATINGS_FILE = "ratings.json"
LOG_PATH = storage.LEGACY["rating_log"]   # the old log, read only by `storage migrate`; who_rated_what.csv is beside it
NAMED_PATH = model.PROJECT_DIR / "history" / "genomes" / "named.json"
CLIP = re.compile(r"^(baseline|genome_\d{2})$")
RATER = re.compile(r"^[\w .'-]{1,40}$")
NAME = re.compile(r"^[\w .,'!?()&+-]{1,60}$")



def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _write(path, data):
    """A temporary file first, so a crash never leaves half a file; waits while a reader has it open."""
    safe_files.write_json(path, data)


# --------------------------------------------------------------------------- #
# The database (storage.py): one run's ratings document <-> rows
# --------------------------------------------------------------------------- #

def _from_db(db, run):
    """The ratings document of `run` as load() returns it, or None when the run is not in the database."""
    head = db.execute("SELECT raters, picked, updated FROM rating_runs WHERE run = ?", (run,)).fetchone()
    if head is None:
        return None
    doc = {"schema_version": SCHEMA_VERSION, "run": run, "scale": [LOWEST, HIGHEST],
           "raters": json.loads(head["raters"]), "scores": {}, "picked": json.loads(head["picked"]),
           "updated": head["updated"], "given": {}, "artifacts": {}}
    for r in db.execute("SELECT * FROM ratings WHERE run = ? ORDER BY rowid", (run,)):
        artifact = None if r["artifact"] is None else bool(r["artifact"])
        doc["scores"].setdefault(r["rater"], {})[r["clip"]] = r["score"]
        doc["given"].setdefault(r["rater"], {})[r["clip"]] = {
            "first_given": r["first_given"], "changes": r["changes"], "score": r["score"], "time": r["time"],
            "account": r["account"], "submitted_by": r["submitted_by"], "artifact": artifact}
        if artifact is not None:
            doc["artifacts"].setdefault(r["rater"], {})[r["clip"]] = artifact
    return doc


def _store(db, run_dir, ratings):
    """Write one run's ratings document into the database (inside the caller's transaction)."""
    run = Path(run_dir).name
    db.execute("INSERT INTO rating_runs (run, folder, raters, picked, updated) VALUES (?, ?, ?, ?, ?) "
               "ON CONFLICT (run) DO UPDATE SET folder = excluded.folder, raters = excluded.raters, "
               "picked = excluded.picked, updated = excluded.updated",
               (run, str(run_dir), json.dumps(ratings.get("raters", []), ensure_ascii=False),
                json.dumps(ratings.get("picked", []), ensure_ascii=False), ratings.get("updated")))
    given, said = ratings.get("given", {}), ratings.get("artifacts", {})
    for rater, clips in ratings.get("scores", {}).items():
        for clip, score in clips.items():
            d = given.get(rater, {}).get(clip, {})
            artifact = said.get(rater, {}).get(clip, d.get("artifact"))
            db.execute("INSERT INTO ratings (run, rater, clip, score, artifact, first_given, time, changes, account, "
                       "submitted_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (run, rater, clip) DO UPDATE SET "
                       "score = excluded.score, artifact = excluded.artifact, first_given = excluded.first_given, "
                       "time = excluded.time, changes = excluded.changes, account = excluded.account, "
                       "submitted_by = excluded.submitted_by",
                       (run, rater, clip, score, None if artifact is None else int(artifact), d.get("first_given"),
                        d.get("time"), d.get("changes", 0), d.get("account"), d.get("submitted_by")))


def _log(db, entries):
    """Add entries to the rating log (it only grows)."""
    db.executemany("INSERT INTO rating_log (time, run, rater, account, clip, score, entry) VALUES (?, ?, ?, ?, ?, ?, ?)",
                   [(e["time"], e["run"], e["rater"], e.get("account"), e["clip"], e.get("score"),
                     json.dumps(e, ensure_ascii=False)) for e in entries])


def _copy_to_folder(run_dir, ratings):
    """The run folder's ratings.json, a copy for reading; the database stays right if it cannot be written."""
    try:
        _write(Path(run_dir) / RATINGS_FILE, ratings)
    except OSError:
        pass


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
    with storage.connect() as db:
        stored = _from_db(db, Path(run_dir).name)
    if stored is not None:
        return stored
    path = Path(run_dir) / RATINGS_FILE
    if path.is_file():
        return safe_files.read_json(path)          # not in the database yet: stored there on its first change
    return {"schema_version": SCHEMA_VERSION, "run": Path(run_dir).name, "scale": [LOWEST, HIGHEST],
            "raters": [], "scores": {}, "picked": [], "updated": None}


TEST_RATER = re.compile(r"^test[ _]user[ _]\d+$", re.IGNORECASE)   # "Test user 3" (account test_user_3)


def is_test_rater(name):
    """A listener name from an account made to try the website (Test user N), whose scores are not evidence."""
    return bool(TEST_RATER.match(str(name or "").strip()))


def evidence(ratings):
    """A run's ratings without the test accounts' scores: what the MOS, the reports, the guidance model and
    the MLflow import count. The website still shows a test account its own scores (load)."""
    keep = [r for r in ratings.get("raters", []) if not is_test_rater(r)]
    return {**ratings, "raters": keep,
            "scores": {r: s for r, s in ratings.get("scores", {}).items() if not is_test_rater(r)},
            **({"artifacts": {r: a for r, a in ratings["artifacts"].items() if not is_test_rater(r)}}
               if isinstance(ratings.get("artifacts"), dict) else {})}


def clean_artifact(value):
    """Whether the listener heard an artifact: True or False (also "yes"/"no", 1/0)."""
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("yes", "no", "true", "false"):
        return value.strip().lower() in ("yes", "true")
    raise ValueError(f"An artifact answer must be true (heard one) or false (none), not {value!r}.")


def is_rated(run_dir):
    return any(load(run_dir)["scores"].values())


def who_path():
    return LOG_PATH.parent / "who_rated_what.csv"


def save_scores(run_dir, scores, submitted_by=None, account=None, artifacts=None):
    """
    Save the scores of one or more listeners: {rater: {clip: score}}, clip being
    "baseline" or "genome_NN". A score given again replaces the earlier one.
    artifacts: {rater: {clip: True/False}}, whether that listener heard an
    artifact in the clip; it may come with a score or later, for a clip the
    listener has already scored. A new or changed score or answer is also
    appended to the review log (sending the same again adds nothing). At most
    MAX_RATERS listeners per run.
    submitted_by: who sent them (the API key's owner or the account's name);
    account: the login account's username, when it was an account.
    """
    run_dir = Path(run_dir)
    run = genome_search.load_run(run_dir)
    rendered = {"baseline"} if run["baseline"].get("audio") else set()
    rendered |= {clip_key(g["number"]) for g in run["genomes"] if g.get("audio")}

    def check_clip(clip):
        if not CLIP.match(str(clip)) or clip not in rendered:
            raise ValueError(f"No rendered clip {clip!r} in this run.")

    cleaned, heard = {}, {}
    for rater, clips in scores.items():
        rater = clean_rater(rater)
        for clip, value in clips.items():
            check_clip(clip)
            cleaned.setdefault(rater, {})[clip] = clean_score(value)
    for rater, clips in (artifacts or {}).items():
        rater = clean_rater(rater)
        for clip, value in clips.items():
            check_clip(clip)
            heard.setdefault(rater, {})[clip] = clean_artifact(value)
    if not cleaned and not heard:
        return load(run_dir)

    with storage.connect(write=True) as db:            # one writer at a time, also across processes
        ratings = load(run_dir)
        raters = list(ratings["raters"])
        for rater in cleaned:
            if rater not in raters:
                raters.append(rater)
        if len(raters) > MAX_RATERS:
            raise ValueError(f"At most {MAX_RATERS} listeners can rate one run; it already has "
                             f"{', '.join(ratings['raters'])}.")
        for rater, clips in heard.items():
            for clip in clips:
                if clip not in cleaned.get(rater, {}) and clip not in ratings["scores"].get(rater, {}):
                    raise ValueError(f"Score {clip} first: an artifact answer is kept with a score.")
        ratings["raters"] = raters
        time = _now()
        log = []
        genomes = {clip_key(g["number"]): g for g in run["genomes"]}
        given = ratings.setdefault("given", {})
        answers = ratings.setdefault("artifacts", {})
        for rater in dict.fromkeys(list(cleaned) + list(heard)):
            before = ratings["scores"].setdefault(rater, {})
            said = answers.setdefault(rater, {})
            for clip in dict.fromkeys(list(cleaned.get(rater, {})) + list(heard.get(rater, {}))):
                previous, previous_artifact = before.get(clip), said.get(clip)
                score = cleaned.get(rater, {}).get(clip, previous)
                artifact = heard.get(rater, {}).get(clip, previous_artifact)
                if previous == score and previous_artifact == artifact and clip in given.get(rater, {}):
                    continue                       # the page sent it again: nothing changed
                before[clip] = score
                if artifact is not None:
                    said[clip] = artifact
                detail = given.setdefault(rater, {}).get(clip) or {"first_given": time, "changes": -1}
                detail.update(score=score, artifact=artifact, time=time, account=account or detail.get("account"),
                              submitted_by=submitted_by, changes=detail["changes"] + 1)
                given[rater][clip] = detail
                genome = genomes.get(clip)
                log.append({"time": time, "run": run["id"], "batch": run.get("batch"), "pile": run.get("pile"),
                            "voice": run["voice"], "origin": run.get("origin"), "text": run["text"],
                            "rater": rater, "account": account, "clip": clip,
                            "genome": genome["number"] if genome else None, "score": score, "previous": previous,
                            "artifact": artifact, "previous_artifact": previous_artifact,
                            "markup": genome["markup"] if genome else run["text"],
                            "speed": genome["speed"] if genome else 1.0,
                            "genes": genome["genes"] if genome else [],
                            "verdict": genome["verdict"] if genome else "baseline",
                            "submitted_by": submitted_by})
        if not log:
            return ratings
        ratings["updated"] = time
        _store(db, run_dir, ratings)
        _log(db, log)
    _copy_to_folder(run_dir, ratings)
    try:
        write_current(who_path(), run_dir.parent)
    except OSError:
        pass                                       # open in a spreadsheet, say; rebuilt on the next save
    return ratings


def artifact_of(ratings, rater, clip):
    """What `rater` said about an artifact in `clip`: True, False, or None (not asked or not answered)."""
    return ratings.get("artifacts", {}).get(rater, {}).get(clip)


def mos(ratings, clip):
    """
    {"mos", "n", "low", "high", "scores": {rater: score}, "artifact_heard", "artifact_answers"} of one clip,
    or None when nobody rated it. artifact_heard: how many of its listeners heard an artifact, out of the
    artifact_answers who answered. Test accounts' scores (is_test_rater) are not counted: they tried the
    website, they did not listen for the study.
    """
    given = {rater: clips[clip] for rater, clips in ratings["scores"].items() if clip in clips and not is_test_rater(rater)}
    if not given:
        return None
    values = list(given.values())
    said = [artifact_of(ratings, rater, clip) for rater in given]
    return {"mos": round(sum(values) / len(values), 2), "n": len(values), "low": min(values), "high": max(values),
            "scores": given, "artifact_heard": sum(1 for a in said if a is True),
            "artifact_answers": sum(1 for a in said if a is not None)}


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

def collection(runs, rater=None, others=True):
    """
    Every rendered clip of several runs (a listening pile, say), for one listener's
    rating page and ranking. `runs` is [(run folder, run dict)].

    Returns {"runs": [...], "clips": [...]}. Each run has its id, voice, pile,
    line, how many clips it has and how many `rater` scored. Each clip has its
    run, "baseline" or the genome's number, what the genome does, its verdict,
    `rater`'s score ("mine", or None) and artifact answer ("mine_artifact": True,
    False or None), the MOS over every listener, and both
    against the same run's baseline, because a score means most next to the
    plain render of the same voice and line. others=False (a listener rating
    blind): no MOS and no other listener's name, only `rater`'s own scores.
    """
    out_runs, clips = [], []
    for folder, run in runs:
        ratings = load(folder)
        own = ratings["scores"].get(rater, {}) if rater else {}
        own_on = ratings.get("given", {}).get(rater, {}) if rater else {}
        if not others:
            ratings = {**ratings, "raters": [rater] if own else [], "scores": {rater: own} if own else {},
                       "artifacts": {rater: ratings.get("artifacts", {}).get(rater, {})}}
        base_mos, base_mine = mos(ratings, "baseline"), own.get("baseline")
        rendered = [("baseline", None)] if run["baseline"].get("audio") else []
        rendered += [(clip_key(g["number"]), g) for g in run["genomes"] if g.get("audio")]
        for key, genome in rendered:
            score, mine = mos(ratings, key), own.get(key)
            clips.append({
                "run": run["id"], "voice": run["voice"], "pile": run.get("pile"), "clip": key,
                "audio": f"{key}.wav", "genome": genome["number"] if genome else None,
                "described": genome["described"] if genome else ["the line with no genes"],
                "markup": genome["markup"] if genome else run["text"],
                "speed": genome["speed"] if genome else 1.0,
                "verdict": genome["verdict"] if genome else "baseline",
                "mine": mine,
                "mine_artifact": artifact_of(ratings, rater, key) if rater else None,
                "mine_on": (own_on.get(key) or {}).get("time") or (ratings.get("updated") if mine is not None else None),
                "mine_above_baseline": round(mine - base_mine, 2)
                if genome and mine is not None and base_mine is not None else None,
                "mos": score,
                "mos_above_baseline": round(score["mos"] - base_mos["mos"], 2) if genome and score and base_mos else None,
            })
        out_runs.append({"id": run["id"], "voice": run["voice"], "pile": run.get("pile"), "batch": run.get("batch"),
                         "text": run["text"], "clips": len(rendered), "mine": sum(1 for k, _ in rendered if k in own),
                         "raters": ratings["raters"]})
    return {"runs": out_runs, "clips": clips}


def load_named():
    return safe_files.read_json(NAMED_PATH, default={"genomes": []})


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

    with storage.connect(write=True) as db:            # also keeps two namings of named.json apart
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
        _store(db, run_dir, ratings)
    _copy_to_folder(run_dir, ratings)
    return entry


def unpick(run_dir, number):
    """Take the name off a genome again (in the run and in the library)."""
    run_dir = Path(run_dir)
    run_id = genome_search.load_run(run_dir)["id"]
    with storage.connect(write=True) as db:
        library = load_named()
        library["genomes"] = [e for e in library["genomes"] if not (e["run"] == run_id and e["genome"] == int(number))]
        _write(NAMED_PATH, library)
        ratings = load(run_dir)
        ratings["picked"] = [p for p in ratings["picked"] if p["genome"] != int(number)]
        _store(db, run_dir, ratings)
    _copy_to_folder(run_dir, ratings)


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
    """Every score ever given or changed, oldest first (the rating_log table)."""
    with storage.connect() as db:
        return [json.loads(r[0]) for r in db.execute("SELECT entry FROM rating_log ORDER BY id")]


CURRENT_FIELDS = ["rater", "account", "batch", "pile", "voice", "run", "clip", "genome", "score", "artifact",
                  "above_plain_line", "rated_on", "first_given", "changes", "verdict", "markup", "text"]


def current_scores(runs_dir=None):
    """
    Who rated what, now: one row per listener and clip, with the score, whether
    they heard an artifact ("yes", "no", or empty when not asked), the date
    it was last given (and first given), the login account, and the score minus
    the same listener's score for the plain line of the same run. Sorted by
    listener, then batch, pile and voice.
    """
    rows = []
    for folder, run in genome_search.list_runs(runs_dir):
        if not (folder / RATINGS_FILE).is_file():
            continue
        ratings = load(folder)
        genomes = {clip_key(g["number"]): g for g in run["genomes"]}
        for rater, clips in ratings["scores"].items():
            detail = ratings.get("given", {}).get(rater, {})
            plain = clips.get("baseline")
            for clip, score in clips.items():
                genome, info = genomes.get(clip), detail.get(clip, {})
                rows.append({
                    "rater": rater, "account": info.get("account"), "batch": run.get("batch"), "pile": run.get("pile"),
                    "voice": run["voice"], "run": run["id"], "clip": clip,
                    "genome": genome["number"] if genome else None, "score": score,
                    "artifact": yes_no(artifact_of(ratings, rater, clip)),
                    "above_plain_line": round(score - plain, 2) if genome and plain is not None else None,
                    "rated_on": info.get("time") or ratings.get("updated"), "first_given": info.get("first_given"),
                    "changes": info.get("changes", 0),
                    "verdict": genome["verdict"] if genome else "baseline",
                    "markup": genome["markup"] if genome else run["text"], "text": run["text"]})
    rows.sort(key=lambda r: (r["rater"].lower(), str(r["batch"]), r["pile"] or 0, r["voice"], r["clip"]))
    return rows


def yes_no(value):
    return "" if value is None else ("yes" if value else "no")


def write_current(path, runs_dir=None):
    rows = current_scores(runs_dir)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = safe_files.temporary_for(path)
    with open(temporary, "w", newline="", encoding="utf-8-sig") as f:    # -sig: Excel reads the accents
        writer = csv.DictWriter(f, fieldnames=CURRENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    safe_files.replace(temporary, path)
    return len(rows)


def who_summary(runs_dir=None):
    """{rater: {"account", "piles": {"batch pile N": [scored, clips]}, "first", "last"}}"""
    totals = {}
    for folder, run in genome_search.list_runs(runs_dir):
        if run.get("batch") and run.get("pile"):
            key = f"{run['batch']} pile {run['pile']}"
            totals[key] = totals.get(key, 0) + (1 if run["baseline"].get("audio") else 0) + \
                sum(1 for g in run["genomes"] if g.get("audio"))
    out = {}
    for row in current_scores(runs_dir):
        one = out.setdefault(row["rater"], {"account": row["account"], "piles": {}, "first": None, "last": None})
        one["account"] = one["account"] or row["account"]
        key = f"{row['batch']} pile {row['pile']}" if row["batch"] else "not in a pile"
        done = one["piles"].setdefault(key, [0, totals.get(key)])
        done[0] += 1
        when = row["rated_on"]
        if when:
            one["first"] = min(one["first"] or when, row["first_given"] or when)
            one["last"] = max(one["last"] or when, when)
    return out


def export_csv(path):
    rows = [{**row, "artifact": yes_no(row.get("artifact")), "previous_artifact": yes_no(row.get("previous_artifact"))}
            for row in read_log()]
    fields = ["time", "rater", "account", "batch", "pile", "voice", "run", "clip", "genome", "score", "previous",
              "artifact", "previous_artifact", "verdict", "speed", "markup", "text", "submitted_by"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review listener ratings (MOS) of genome searches.")
    parser.add_argument("--named", action="store_true", help="list the named genomes")
    parser.add_argument("--export", metavar="FILE", help="write every score ever given or changed (the log) as CSV")
    parser.add_argument("--current", metavar="FILE", help="write the current score of every listener and clip as CSV")
    parser.add_argument("--who", action="store_true", help="who has scored what, per listener and pile")
    args = parser.parse_args()

    if args.current:
        print(f"{write_current(args.current)} scores written to {args.current}")
    elif args.who:
        summary = who_summary()
        if not summary:
            print("Nobody has scored anything yet.")
        for rater, one in summary.items():
            print(f"{rater}  (account {one['account'] or '-'}; from {one['first']} to {one['last']})")
            for pile, (done, total) in one["piles"].items():
                print(f"    {pile}: {done} of {total or '?'} clips scored")
        print(f"\nThe details: {who_path()}")
    elif args.export:
        print(f"{export_csv(args.export)} scores written to {args.export}")
    elif args.named:
        for entry in load_named()["genomes"]:
            score = f"MOS {entry['mos']['mos']:.2f} ({entry['mos']['n']} listeners)" if entry["mos"] else "not rated"
            where = (f"made by hand ({len(entry['genes'])} genes)" if entry.get("edited")
                     else f"genome {entry['genome']} of {entry['run']}")
            print(f"{entry['name']}: {where}, {score}\n    {entry.get('markup') or entry['genes']}")
    else:
        for folder, run, rank in rated_runs():
            best = next((g for g in rank["genomes"] if g["mos"]), None)
            base = f"baseline MOS {rank['baseline']['mos']:.2f}" if rank["baseline"] else "baseline not rated"
            print(f"{run['id']}  ({', '.join(rank['raters'])}; {rank['rated']} genomes rated; {base})")
            if best:
                print(f"    best: genome {best['number']} MOS {best['mos']['mos']:.2f}  {best['markup']}")
