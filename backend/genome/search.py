"""search.py | Find genomes for a voice: try them on a line, and judge each one with the sanity check.

The method for now is a random search, as the first Dramatizer started: draw 10
random genomes with the chosen number of genes each, render the line once
plain (the baseline) and once with each genome, and judge every genome's clip
with reward.sanity_check against the baseline:

    fine       what is heard differs from the baseline by at most 10 % of the
               words (at least 1 word may differ; "uh", "ha" and other fillers
               do not count), it still sounds like the same speaker (similarity
               to the baseline at least 0.88), and it is not much slower than
               its speed gene asks (pace at least 0.75)
    terrible   any one fails

    from backend.genome.search import new_run, run_search

    run = new_run("reference.wav", "the reference's words", text="The line to speak.", genes=3, seed=7)
    run_search(run)          # renders 11 clips; saves after each one

A run is a folder in RUNS_DIR (output/genome_search/, never committed) with
run.json, baseline.wav and genome_01.wav ... genome_10.wav. run.json is written
before anything is rendered and after every clip, so a run that stopped (a
crash, the PC turned off) continues where it was: run_search(run) again.

Every clip of a run uses the same render seed, so the baseline and a genome's
clip differ only by the genome. The genome seed decides which genomes are drawn.

    python -m backend.genome.search REFERENCE_AUDIO "REFERENCE TEXT" --text "The line." --genes 3 --seed 7
    python -m backend.genome.search REFERENCE_AUDIO "REFERENCE TEXT" --prepare-only   # draw, render nothing
    python -m backend.genome.search --resume output/genome_search/RUN_ID
"""
import argparse
import json
import re
import secrets
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from backend import model, safe_files
from backend.genome.candidate import Candidate
from backend.genome.genome_factory import MAX_OPERATIONS, GenomeFactory

RUNS_DIR = model.OUTPUT_DIR / "genome_search"
RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
SCHEMA_VERSION = 1
GENES = 3                    # genes per genome, when none is asked for
DRAW_ATTEMPTS = 50           # per genome wanted, before giving up on finding different ones


@dataclass(frozen=True)
class Method:
    label: str
    genomes: int             # how many genomes it tries
    draw: callable           # (text, genes, count, seed) -> [Candidate]
    about: str


def draw_random(text, genes, count, seed):
    """
    `count` random genomes of exactly `genes` genes, from the first Dramatizer's
    GenomeFactory. Two genomes that would give the same markup on this line are
    one clip, so only genomes with different markup are kept.
    """
    from backend.speech.tts_formatter import process_tts_string

    factory = GenomeFactory(min_operations=genes, max_operations=genes, seed=seed)
    genomes, seen = [], set()

    for _ in range(count * DRAW_ATTEMPTS):
        genome = factory.random_genome(text)
        plan = process_tts_string(text, genome.resolve(text))
        key = (plan.text, plan.speed, plan.instruct)
        if key not in seen:
            seen.add(key)
            genomes.append(genome)
            if len(genomes) == count:
                return genomes

    raise ValueError(f"Found only {len(genomes)} different genomes of {genes} genes for this line; "
                     "use a longer line or fewer genes.")


METHODS = {
    "random": Method("Random search", 10, draw_random,
                     "10 random genomes, each judged against the plain line by the sanity check"),
}


# --------------------------------------------------------------------------- #
# Runs on disk
# --------------------------------------------------------------------------- #


def run_folder(run_id, runs_dir=None):
    """The folder of a run, only for a valid run id inside the runs folder."""
    if not RUN_ID.match(run_id or ""):
        raise ValueError(f"Not a run id: {run_id!r}")
    return Path(runs_dir or RUNS_DIR) / run_id


def load_run(run):
    """The run.json of a run folder (or of a run.json path) as a dict."""
    path = Path(run)
    path = path if path.name == "run.json" else path / "run.json"
    return safe_files.read_json(path)


def save_run(run_dir, run):
    """Written to a temporary file first, so a crash never leaves half a run.json."""
    safe_files.write_json(Path(run_dir) / "run.json", run)     # waits while a reader has it open (Windows)


def list_runs(runs_dir=None):
    """Every run, newest first: [(folder, run dict)]. Folders without a readable run.json are skipped."""
    runs = []
    for path in Path(runs_dir or RUNS_DIR).glob("*/run.json"):
        try:
            runs.append((path.parent, load_run(path)))
        except (OSError, ValueError):
            continue
    return sorted(runs, key=lambda item: item[1].get("created", ""), reverse=True)


def summary(run):
    """How far a run is and how it went: {"done", "total", "fine", "terrible", "failed"}."""
    verdicts = [genome.get("verdict") for genome in run["genomes"]]
    return {
        "done": sum(v is not None for v in verdicts),
        "total": len(verdicts),
        "fine": verdicts.count("fine"),
        "terrible": verdicts.count("terrible"),
        "failed": verdicts.count("failed"),
    }


def _safe_name(text):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)[:40].strip("_") or "voice"


def _plain(value):
    """numpy numbers as plain ones, so they can go in JSON."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def render_seed_default():
    """VOICE_SEED from .env (default 0), or a new one for "random"; the same for every clip of a run."""
    seed = model.setting("VOICE_SEED", "0")
    return secrets.randbelow(2**31) if str(seed).strip().lower() == "random" else int(seed)


# --------------------------------------------------------------------------- #
# Making a run, then running it
# --------------------------------------------------------------------------- #


def new_run(reference_audio, reference_text, text=None, voice=None, method="random", genes=GENES,
            seed=None, render_seed=None, origin=None, runs_dir=None):
    """
    Draw the genomes and write the run, without rendering anything. Returns the run's folder.

    reference_audio  the recording of the voice the search is for
    reference_text   the words spoken in it
    text             the line to speak; without it, the reference's own words (as the first Dramatizer did)
    voice            a name for the run (default: the reference file's name)
    method           a key of METHODS
    genes            genes per genome, from 1 to MAX_OPERATIONS
    seed             which genomes are drawn (default: a new one, saved in the run)
    render_seed      the seed of every render (default: VOICE_SEED)
    origin           where the voice comes from (split_voice_data.voice_origin), saved as it is
    """
    from backend.speech.tts_formatter import process_tts_string
    from backend.metrics.reward import CONFIG

    reference_audio = Path(reference_audio)
    if not reference_audio.is_file():
        raise FileNotFoundError(f"Reference audio not found: {reference_audio}")
    if not reference_text.strip():
        raise ValueError("reference_text is empty: give the words spoken in the reference audio.")
    text = (text or reference_text).strip()
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}: use one of {', '.join(METHODS)}.")
    if not 1 <= int(genes) <= MAX_OPERATIONS:
        raise ValueError(f"Genes per genome must be from 1 to {MAX_OPERATIONS}.")

    genes = int(genes)
    seed = secrets.randbelow(10_000) if seed is None else int(seed)
    render_seed = render_seed_default() if render_seed is None else int(render_seed)
    chosen = METHODS[method]
    voice = voice or reference_audio.stem

    genomes = []
    for number, genome in enumerate(chosen.draw(text, genes, chosen.genomes, seed), start=1):
        plan = process_tts_string(text, genome.resolve(text))
        genomes.append({
            "number": number,
            "genes": genome.operations,
            "described": genome.describe(text),
            "operations": genome.resolve(text),
            "markup": plan.text,
            "speed": plan.speed,
            "warnings": plan.warnings,
            "audio": None,
            "sanity": None,
            "verdict": None,
        })

    created = datetime.now()
    name = f"{_safe_name(voice)}_{created.strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
    # Two runs in the same millisecond (the Windows clock is coarse) get a number after the name
    for number in range(1, 1000):
        run_dir = Path(runs_dir or RUNS_DIR) / (name if number == 1 else f"{name}_{number}")
        try:
            run_dir.mkdir(parents=True)
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError(f"No free run folder for {name}")

    run = {
        "schema_version": SCHEMA_VERSION,
        "id": run_dir.name,
        "method": method,
        "method_label": chosen.label,
        "voice": voice,
        "origin": origin,
        "reference_audio": str(reference_audio),
        "reference_text": reference_text.strip(),
        "text": text,
        "genes_per_genome": genes,
        "seed": seed,
        "render_seed": render_seed,
        "limits": {"wer": CONFIG.sanity_wer_limit, "words": CONFIG.sanity_wer_words,
                   "similarity": CONFIG.sanity_similarity, "similarity_short": CONFIG.sanity_similarity_short,
                   "long_seconds": CONFIG.sanity_long_seconds, "pace": CONFIG.sanity_pace_min},
        "created": created.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "ready",
        "error": None,
        "baseline": {"audio": None, "heard": None},
        "genomes": genomes,
    }
    save_run(run_dir, run)
    return run_dir


def render_plan(reference_audio, reference_text, plan, path, seed):
    """Speak a tts_formatter plan in the reference's voice with CosyVoice 3, and save it as a wav."""
    import torchaudio

    from backend.speech.Voice_model import set_seed
    from backend.speech.tts_formatter import synthesize

    cosyvoice = model.voice_model()
    set_seed(seed)
    audio = synthesize(cosyvoice, plan, str(reference_audio), reference_text)
    torchaudio.save(str(path), audio.cpu(), cosyvoice.sample_rate)
    return path


def judge_with_sanity(reference_audio, baseline_audio, test_audio, baseline_heard=None, speed=1.0):
    """reward.sanity_check, the judge of every genome."""
    from backend.metrics.reward import sanity_check
    return sanity_check(reference_audio, baseline_audio, test_audio, baseline_heard=baseline_heard, speed=speed)


def run_search(run_dir, render=render_plan, judge=judge_with_sanity, lock=None, on_progress=None):
    """
    Render what is not rendered yet and judge what is not judged yet, saving after each clip.

    render       (reference_audio, reference_text, plan, path, seed) -> path
    judge        (reference_audio, baseline_audio, test_audio, baseline_heard, speed) -> sanity_check's dict
    lock         held around each render and judgement, so a web page can share the GPU
    on_progress  called with the run dict after every clip

    Returns the finished run dict. A genome whose render or judgement fails is
    marked "failed" with its error, and the search goes on with the next one.
    """
    from backend.speech.tts_formatter import process_tts_string

    run_dir = Path(run_dir)
    run = load_run(run_dir)
    run.update(status="running", error=None)
    save_run(run_dir, run)
    lock = lock or nullcontext()
    reference_audio = run["reference_audio"]

    def saved():
        save_run(run_dir, run)
        if on_progress:
            on_progress(run)

    try:
        baseline = run_dir / "baseline.wav"
        if run["baseline"]["audio"] is None or not baseline.is_file():
            started = time.time()
            with lock:
                render(reference_audio, run["reference_text"], process_tts_string(run["text"], []), baseline,
                       run["render_seed"])
            run["baseline"].update(audio=baseline.name, seconds=round(time.time() - started, 1))
            saved()

        for genome in run["genomes"]:
            if genome["verdict"] is not None:
                continue
            path = run_dir / f"genome_{genome['number']:02d}.wav"
            started = time.time()
            try:
                with lock:
                    if genome["audio"] is None or not path.is_file():
                        plan = process_tts_string(run["text"], genome["operations"])
                        render(reference_audio, run["reference_text"], plan, path, run["render_seed"])
                        genome["audio"] = path.name
                    result = _plain(judge(reference_audio, baseline, path, run["baseline"]["heard"], genome["speed"]))
                run["baseline"]["heard"] = run["baseline"]["heard"] or result.get("baseline_heard")
                genome.update(sanity=result, verdict="fine" if result["acceptable"] else "terrible",
                              seconds=round(time.time() - started, 1), error=None)
            except Exception as error:
                genome.update(verdict="failed", error=f"{type(error).__name__}: {error}")
            saved()

        run["status"] = "finished"
        run["finished"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except BaseException as error:
        run.update(status="stopped", error=f"{type(error).__name__}: {error}")
        save_run(run_dir, run)
        raise

    save_run(run_dir, run)
    return run


def print_run(run):
    print(f"\n{run['method_label']} for {run['voice']}: {run['genes_per_genome']} genes per genome, "
          f"genome seed {run['seed']}, render seed {run['render_seed']}")
    print(f"Line: {run['text']}")
    for genome in run["genomes"]:
        verdict = genome["verdict"] or "not judged"
        numbers = ""
        if genome.get("sanity"):
            numbers = f"  WER {genome['sanity']['wer']:.3f}, similarity {genome['sanity']['similarity']:.3f}"
        speed = f"  (speed {genome['speed']:g})" if genome["speed"] != 1 else ""
        print(f"  {genome['number']:2d}. {verdict:10s} {genome['markup']}{speed}{numbers}")
    counts = summary(run)
    print(f"{counts['done']} of {counts['total']} judged: {counts['fine']} fine, {counts['terrible']} terrible"
          + (f", {counts['failed']} failed" if counts["failed"] else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find genomes for a voice, judged by the sanity check.")
    parser.add_argument("reference_audio", nargs="?", help="a recording of the voice")
    parser.add_argument("reference_text", nargs="?", help="the words spoken in it")
    parser.add_argument("--text", help="the line to speak (default: the reference's own words)")
    parser.add_argument("--voice", help="a name for the run (default: from the reference file)")
    parser.add_argument("--method", default="random", choices=list(METHODS))
    parser.add_argument("--genes", type=int, default=GENES, help=f"genes per genome, 1 to {MAX_OPERATIONS}")
    parser.add_argument("--seed", type=int, help="which genomes are drawn (default: a new one)")
    parser.add_argument("--render-seed", type=int, help="the seed of every render (default: VOICE_SEED)")
    parser.add_argument("--prepare-only", action="store_true", help="draw the genomes and save the run, render nothing")
    parser.add_argument("--resume", help="a run folder to continue")
    args = parser.parse_args()

    if args.resume:
        folder = Path(args.resume)
    else:
        if not (args.reference_audio and args.reference_text):
            parser.error("give REFERENCE_AUDIO and REFERENCE_TEXT, or --resume RUN_FOLDER")
        folder = new_run(args.reference_audio, args.reference_text, args.text, args.voice, args.method,
                         args.genes, args.seed, args.render_seed)
        print(f"Run saved in {folder}")

    if args.prepare_only:
        print_run(load_run(folder))
        print(f"\nNothing rendered. To render and judge: python -m backend.genome.search --resume \"{folder}\"")
    else:
        print_run(run_search(folder, on_progress=lambda run: print(f"  {summary(run)['done']} judged", flush=True)))
