"""report_eval.py | The "Report evaluation": the paper's blind listening test on the locked test voices.

    python main_backend.py report-eval --prepare      # choose voices, items, piles and orders; render nothing
    python main_backend.py report-eval                # prepare (once), render until done, publish, assign
    python main_backend.py report-eval --status       # how far the renders are
    python main_backend.py report-eval --publish-only # publish and assign what is rendered

Unlocked on 18 Nov 2026 for this evaluation only: the test voices (voice-split-v2) were never heard before.
They are used here once; nothing is tuned on them afterwards.

THE DESIGN (fixed in plan.json before anything is rendered; the paper's Method section describes it)

  Voices     VOICES_PER_DATASET test voices from each dataset (CREMA-D, LibriTTS-R, VCTK), 12 in all: in each
             dataset the voices are split at the median pitch of their reference clip, and half are drawn at
             random (seeded) from the lower and half from the higher group.
  Sentences  SENTENCES: four lines never used in development, a short, two medium and a long one, each written
             for a different situation (anger, hurt, sarcasm, fear), so a delivery has something to carry.
  Items      every voice x every sentence = 48 items. Each item is rendered three ways, the CONDITIONS:
               plain     the line with no genome
               random    a random genome from the current gene set, with no checks at all
               guarded   the Dramatizer's guarded draw: only the gene types in ALLOWED (chosen from the pilot by
                         ALLOWED_RULE), the same number of genes, and it must pass the sanity check (words,
                         voice, pace) and the audio check (artifact_check.py); up to GUARD_TRIES draws
             Both genomes of an item have the same number of genes, 1, 2 or 3, balanced over voices and
             sentences ((voice + 2 x sentence) mod 3). All three clips of an item share one render seed, so
             they differ only by the genome.
  Listeners  LISTENERS (Ash, Julius, Min, Rama), shuffled (seeded) into the letters A-D. There are six pairs of
             listeners; each pair is one pile of 8 items (24 clips), item -> pile (voice + sentence) mod 6, so
             every item is heard by 2 listeners, every two listeners share 8 items, every listener hears 24 items
             (72 clips), each sentence 6 times, and no listener hears an item twice.
  Blindness  The run.json a listener's page reads holds no genes, no markup, no condition and no "plain line"
             label: three versions, genome_01 to genome_03. Which is which is in key.json, outside the runs
             folder, which no API route serves. The order of the three versions follows the six possible
             orders in turn, so each condition is first, second and third equally often. "blind": nobody sees
             another listener's scores; "only": just the pile's two listeners may open or rate it.

Everything except the published runs goes to OUTPUT_DIR/report_evaluation/<batch>/ (never committed):
plan.json, items/<item>/state.json with every render and check, and key.json.
"""
import argparse
import hashlib
import itertools
import json
import random
import shutil
import statistics as st
import sys
import time
from datetime import datetime
from pathlib import Path

from backend import model, safe_files
from backend.genome import artifact_check, search
from backend.genome.genome_factory import GenomeFactory

BATCH = "report-evaluation"
LABEL = "Report evaluation"
OUT_DIR = model.OUTPUT_DIR / "report_evaluation"
SEED = 20261118
DATASETS = ("CREMA-D", "LibriTTS-R", "VCTK")
VOICES_PER_DATASET = 4
SENTENCES = [
    {"id": "s1", "length": "short", "situation": "anger", "text": "Don't you dare walk away from me."},
    {"id": "s2", "length": "medium", "situation": "hurt",
     "text": "I waited all night by the phone, and you never even called."},
    {"id": "s3", "length": "medium", "situation": "sarcasm",
     "text": "Oh, wonderful, another meeting that could have been an email."},
    {"id": "s4", "length": "long", "situation": "fear",
     "text": "When the lights went out in the old house, I heard footsteps on the stairs, slow and heavy, "
             "and then someone whispered my name."},
]
CONDITIONS = ("plain", "random", "guarded")
GENE_COUNTS = (1, 2, 3)
GUARD_TRIES = 4
DRAW_ATTEMPTS = 2000
LISTENERS = {"Ash": "ash", "Julius": "julius", "Min": "min", "Rama": "rama"}
# From the pilot (mos-diverse-v1, evidence_charts.gene_type_table, 18 Nov 2026): a gene type is allowed when its
# share of genomes rated above the plain line is at least the overall share (31 %) and at most one of the three
# listeners scored it below their own average. Frozen here so the guarded condition does not change later.
ALLOWED_RULE = ("pilot share above the plain line >= the overall 31 %, and at most 1 of 3 listeners below their "
                "own average (evidence_charts.gene_type_table on mos-diverse-v1, 18 Nov 2026)")
ALLOWED = ["accent", "laughing speech", "lipsmack", "mn", "noise", "pause (realization)", "quick breath", "sigh"]


def _save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_files.write_json(path, data)


def _load(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def batch_dir(batch=BATCH):
    return OUT_DIR / batch


def safe(name):
    return name.replace("/", "_")


def gene_type(gene):
    """A gene's type in words, as the pilot's evidence names it ("pause (realization)", "laughing speech")."""
    from backend.rag.corpus import gene_name
    return gene_name(gene)


# ---- the design -------------------------------------------------------------------------------------

def choose_voices(test_voices, pitch, per_dataset=VOICES_PER_DATASET, seed=SEED):
    """Per dataset, half the voices from below its median pitch and half from above, drawn at random (seeded)."""
    rng = random.Random(seed)
    chosen = []
    for dataset in DATASETS:
        names = sorted(v for v in test_voices if v.startswith(dataset + "/") and pitch.get(v) is not None)
        if len(names) < per_dataset:
            raise ValueError(f"{dataset} has only {len(names)} test voices with a pitch; {per_dataset} are needed.")
        middle = st.median(pitch[v] for v in names)
        low = [v for v in names if pitch[v] <= middle]
        high = [v for v in names if pitch[v] > middle]
        chosen += rng.sample(low, per_dataset // 2) + rng.sample(high, per_dataset - per_dataset // 2)
    return chosen


def design(voices, listeners=tuple(LISTENERS), seed=SEED):
    """Items, piles and listener letters for the voices (in dataset round-robin order). Pure: no files."""
    rng = random.Random(seed)
    by_dataset = {d: [v for v in voices if v.startswith(d + "/")] for d in DATASETS}
    for names in by_dataset.values():
        rng.shuffle(names)
    order = [v for group in itertools.zip_longest(*by_dataset.values()) for v in group if v]   # CREMA, Libri, VCTK, ...
    people = list(listeners)
    rng.shuffle(people)
    letters = dict(zip("ABCD", people))
    pairs = list(itertools.combinations("ABCD", 2))                  # pile 1..6 = AB, AC, AD, BC, BD, CD
    orders = list(itertools.permutations(CONDITIONS))
    items = []
    for v, voice in enumerate(order):
        for s, sentence in enumerate(SENTENCES):
            pile = (v + s) % len(pairs) + 1
            items.append({"item": f"{safe(voice)}_{sentence['id']}", "voice": voice, "sentence": sentence["id"],
                          "genes": GENE_COUNTS[(v + 2 * s) % len(GENE_COUNTS)], "pile": pile,
                          "listeners": [letters[x] for x in pairs[pile - 1]]})
    for pile in range(1, len(pairs) + 1):                            # each pile walks through the six orders
        for i, item in enumerate(x for x in items if x["pile"] == pile):
            item["order"] = list(orders[(i + pile) % len(orders)])
    return {"voice_order": order, "letters": letters, "piles": {str(i + 1): [letters[a], letters[b]]
                                                                  for i, (a, b) in enumerate(pairs)}, "items": items}


def balance(d):
    """Counts that show the design is balanced (checked by the tests, printed by --prepare)."""
    per_listener = {}
    for name in d["letters"].values():
        mine = [i for i in d["items"] if name in i["listeners"]]
        per_listener[name] = {"items": len(mine), "clips": 3 * len(mine), "voices": len({i["voice"] for i in mine}),
                              "per_sentence": {s["id"]: sum(i["sentence"] == s["id"] for i in mine) for s in SENTENCES},
                              "per_gene_count": {str(n): sum(i["genes"] == n for i in mine) for n in GENE_COUNTS}}
    first = {c: sum(i["order"][0] == c for i in d["items"]) for c in CONDITIONS}
    shared = {f"{a}+{b}": sum(a in i["listeners"] and b in i["listeners"] for i in d["items"])
              for a, b in itertools.combinations(sorted(d["letters"].values()), 2)}
    return {"items": len(d["items"]), "per_listener": per_listener, "condition_first": first, "items_shared": shared}


def prepare(batch=BATCH):
    """plan.json and every item's state.json; made once, then reused (the design never changes after this)."""
    from backend.genome.explore import acoustics
    from backend.voice_data import split_voice_data
    from backend.voice_data.Voice_Importer import get_clip

    folder = batch_dir(batch)
    plan = _load(folder / "plan.json")
    if plan:
        return plan
    chosen_clips = dict(split_voice_data.voice_clips("test"))
    clips = {f"{c.dataset}/{c.speaker}": c for c in split_voice_data.read_library()
             if chosen_clips.get(f"{c.dataset}/{c.speaker}") == c.clip_id}
    pitch, refs = {}, {}
    for name, clip in sorted(clips.items()):
        audio, text = get_clip(clip)
        refs[name] = {"reference_audio": str(audio), "reference_text": text, "clip_id": clip.clip_id,
                      "origin": split_voice_data.voice_origin(clip)}
        try:
            pitch[name] = round(acoustics(audio)["pitch_st"], 2)
        except Exception as error:                               # noqa: BLE001 (a voice without pitch is left out)
            print(f"  no pitch for {name}: {error}", flush=True)
    voices = choose_voices(list(refs), pitch)
    d = design(voices)
    plan = {"batch": batch, "label": LABEL, "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "unlocked": "test voices of voice-split-v2, unlocked 18 Nov 2026 for this evaluation only",
            "seed": SEED, "render_seed": search.render_seed_default(), "sentences": SENTENCES,
            "conditions": list(CONDITIONS), "guard_tries": GUARD_TRIES, "allowed": ALLOWED, "allowed_rule": ALLOWED_RULE,
            "listeners": LISTENERS, "voices": [{"voice": v, "pitch_st": pitch[v], **refs[v]} for v in voices],
            "pitch_of_every_test_voice": pitch, **d, "balance": balance(d)}
    for item in d["items"]:
        _save(folder / "items" / item["item"] / "state.json", {**item, "renders": {}})
    _save(folder / "plan.json", plan)
    return plan


# ---- rendering ----------------------------------------------------------------------------------------

def draw(text, genes, seed, allowed=None):
    """A random genome of exactly `genes` genes on `text` (from the current gene set); with `allowed`, only
    gene types in it. Returns (genes, operations)."""
    factory = GenomeFactory(min_operations=genes, max_operations=genes, seed=seed)
    for _ in range(DRAW_ATTEMPTS):
        genome = factory.random_genome(text)
        if allowed is None or all(gene_type(g) in allowed for g in genome.operations):
            return genome.operations, genome.resolve(text)
    raise ValueError(f"No genome of {genes} genes with only {allowed} found on {text!r}.")


def stable_seed(*parts):
    """A seed from the parts that is the same in every Python process (hash() of a string is not)."""
    return int(hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()[:8], 16)


def _tts(text, operations):
    from backend.speech.tts_formatter import process_tts_string
    return process_tts_string(text, operations)


def render_item(plan, item_dir, render=search.render_plan, judge=search.judge_with_sanity):
    """Render one item's three conditions (resuming); saves after each render. Returns its state."""
    state = _load(item_dir / "state.json")
    voice = next(v for v in plan["voices"] if v["voice"] == state["voice"])
    text = next(s["text"] for s in plan["sentences"] if s["id"] == state["sentence"])
    seed = plan["render_seed"]
    r = state["renders"]

    def one(name, genes, operations):
        path = item_dir / f"{name}.wav"
        tts = _tts(text, operations)
        started = time.time()
        render(voice["reference_audio"], voice["reference_text"], tts, path, seed)
        out = {"audio": path.name, "genes": genes, "operations": operations, "markup": tts.text, "speed": tts.speed,
               "warnings": tts.warnings, "seconds": None}
        if name != "plain":
            sanity = search._plain(judge(voice["reference_audio"], item_dir / "plain.wav", path,
                                         r["plain"].get("heard"), tts.speed))
            r["plain"]["heard"] = r["plain"].get("heard") or sanity.get("baseline_heard")
            audio = artifact_check.check_file(path, r["plain"]["audio_check"].get("peak_db"))
            out.update(sanity=sanity, audio_check=audio, passed=bool(sanity["acceptable"] and audio["ok"]))
        else:
            out.update(audio_check=artifact_check.check_file(path), heard=None)
        out["seconds"] = round(time.time() - started, 1)
        return out

    if "plain" not in r:
        r["plain"] = one("plain", [], [])
        _save(item_dir / "state.json", state)
    if "random" not in r:
        genes, ops = draw(text, state["genes"], seed=stable_seed(SEED, state["item"], "random"))
        r["random"] = one("random", genes, ops)
        _save(item_dir / "state.json", state)
    tries = r.setdefault("guarded_tries", [])
    while "guarded" not in r:
        n = len(tries)
        genes, ops = draw(text, state["genes"], seed=stable_seed(SEED, state["item"], "guarded", n),
                          allowed=set(plan["allowed"]))
        attempt = one(f"guarded_try{n + 1}", genes, ops)
        tries.append(attempt)
        if attempt["passed"] or len(tries) >= plan["guard_tries"]:
            r["guarded"] = {**attempt, "try": len(tries), "guard_failed": not attempt["passed"]}
        _save(item_dir / "state.json", state)
    return state


def render_all(plan, hours=None):
    folder = batch_dir(plan["batch"])
    deadline = time.time() + hours * 3600 if hours else None
    started = time.time()
    for n, item in enumerate(plan["items"], 1):
        if deadline and time.time() > deadline:
            print("Out of time: stopping before the next item.", flush=True)
            return False
        item_dir = folder / "items" / item["item"]
        t = time.time()
        state = render_item(plan, item_dir)
        g = state["renders"]["guarded"]
        print(f"  {n:2d}/{len(plan['items'])} {item['item']:28s} guarded after {g['try']} tr"
              f"{'y' if g['try'] == 1 else 'ies'}{' (FAILED the guard)' if g['guard_failed'] else ''}"
              f"  {time.time() - t:5.1f} s", flush=True)
    print(f"All rendered ({(time.time() - started) / 60:.1f} min this session).", flush=True)
    return True


# ---- publishing ---------------------------------------------------------------------------------------

def publish(plan, runs_dir=None):
    """One masked run per item in RUNS_DIR, and key.json beside the plan. Never over a run that has ratings."""
    from backend.genome import ratings

    folder = batch_dir(plan["batch"])
    texts = {s["id"]: s["text"] for s in plan["sentences"]}
    key, published = {}, []
    for item in plan["items"]:
        state = _load(folder / "items" / item["item"] / "state.json")
        if "guarded" not in state["renders"]:
            continue
        run_id = f"{plan['batch']}_p{item['pile']}_{item['item']}"
        run_dir = search.run_folder(run_id, runs_dir)
        versions = {}
        for number, condition in enumerate(item["order"], 1):
            versions[f"genome_{number:02d}"] = condition
        key[run_id] = {"item": item["item"], "voice": item["voice"], "sentence": item["sentence"],
                       "pile": item["pile"], "genes": item["genes"], "versions": versions,
                       "renders": {c: state["renders"][c] for c in CONDITIONS},
                       "guarded_tries": len(state["renders"].get("guarded_tries", []))}
        if run_dir.exists() and ratings.is_rated(run_dir):
            published.append(run_id)
            continue
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        genomes = []
        for clip, condition in versions.items():
            number = int(clip[-2:])
            shutil.copyfile(folder / "items" / item["item"] / state["renders"][condition]["audio"], run_dir / f"{clip}.wav")
            genomes.append({"number": number, "genes": [], "described": [f"Version {number} of the line"],
                            "operations": [], "markup": texts[item["sentence"]], "speed": 1.0, "warnings": [],
                            "audio": f"{clip}.wav", "verdict": "fine", "sanity": None})
        voice = next(v for v in plan["voices"] if v["voice"] == item["voice"])
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        only = sorted(plan["listeners"][name] for name in item["listeners"])
        search.save_run(run_dir, {
            "schema_version": search.SCHEMA_VERSION, "id": run_id, "method": "report-evaluation",
            "method_label": LABEL, "voice": item["voice"], "origin": voice["origin"], "text": texts[item["sentence"]],
            "reference_audio": voice["reference_audio"], "reference_text": voice["reference_text"],
            "genes_per_genome": "hidden", "seed": plan["seed"], "render_seed": plan["render_seed"],
            "created": now, "finished": now, "status": "finished", "error": None,
            "batch": plan["batch"], "pile": item["pile"], "blind": True, "sealed": True, "only": only,
            "baseline": {"audio": None, "heard": None}, "genomes": genomes})
        published.append(run_id)
    _save(folder / "key.json", {"batch": plan["batch"], "made": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "about": "Which version is which condition. Do not open before every score is in.",
                                "runs": key})
    return published


def assign(plan):
    """Give each listener the piles they are in (and no other account any)."""
    from backend.api import accounts

    for name, username in plan["listeners"].items():
        for pile, pair in plan["piles"].items():
            if name in pair:
                accounts.assign(username, plan["batch"], int(pile))
    return {u: accounts.get(u).assignments for u in plan["listeners"].values()}


def status(plan):
    folder = batch_dir(plan["batch"])
    states = [_load(folder / "items" / i["item"] / "state.json") for i in plan["items"]]
    done = sum("guarded" in s["renders"] for s in states)
    failed = sum(s["renders"].get("guarded", {}).get("guard_failed", False) for s in states)
    renders = sum(len(s["renders"].get("guarded_tries", [])) + ("plain" in s["renders"]) + ("random" in s["renders"])
                  for s in states)
    return {"items_done": done, "of": len(states), "renders": renders, "guard_failed": failed}


def main(argv=None):
    parser = argparse.ArgumentParser(description="The Report evaluation: blind listening test on the test voices.")
    parser.add_argument("--batch", default=BATCH)
    parser.add_argument("--prepare", action="store_true", help="choose voices and the design only; render nothing")
    parser.add_argument("--hours", type=float, help="start no new item after this many hours")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--publish-only", action="store_true")
    parser.add_argument("--no-assign", action="store_true", help="publish without giving the listeners their piles")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    plan = prepare(args.batch)
    if args.prepare:
        print(json.dumps({"voices": plan["voice_order"], "letters": plan["letters"], "piles": plan["piles"],
                          "balance": plan["balance"]}, indent=1, ensure_ascii=False))
        return
    if args.status:
        print(status(plan))
        return
    if not args.publish_only and not render_all(plan, args.hours):
        print(status(plan))
        return
    published = publish(plan)
    print(f"Published {len(published)} runs in {search.RUNS_DIR}; the key is in {batch_dir(args.batch) / 'key.json'}")
    if not args.no_assign:
        for user, piles in assign(plan).items():
            print(f"  {user}: {', '.join(p for p in piles if p.startswith(args.batch))}")
    print(status(plan))


if __name__ == "__main__":
    main()
