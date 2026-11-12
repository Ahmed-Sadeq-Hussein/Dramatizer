"""editor.py | Pick a genome, edit its genes by hand, save it under a name, and render it in a voice.

    from backend.genome import editor

    genes = editor.clean_genes([{"feature": "laughter", "position": 0.3}, {"feature": "speed", "rate": 1.1}])
    editor.preview("Well, you really did it.", genes)       # the genes in words, the markup (no GPU)
    editor.save("Giggly", genes, named_by="Julius")         # into the named genomes
    editor.render(reference_audio, reference_text, "A line.", genes, "out.wav", seed=0)   # GPU

The genomes to pick from (library) are the named genomes and every genome of
every search. An edited genome is kept with the named genomes
(history/genomes/named.json) with "run": null and "edited": true, and where it
came from in "based_on". Names are unique across all named genomes, ignoring case.

A gene is checked strictly, because it comes from a web page:

    {"feature": <a targeted feature>, "position": 0..1}
    {"feature": "pause", "position": 0..1, "kind": "dramatic" | "realization"}
    {"feature": "speed", "rate": 0.5..2.0}                at most one

A saved genome has 1 to MAX_GENES genes. A render may have none: that is the
plain line, to compare with.
"""
from datetime import datetime
from pathlib import Path

from backend import safe_files
from backend.genome import ratings, search
from backend.genome.candidate import WORDS, Candidate
from backend.genome.genome_factory import FEATURES, PAUSE_KINDS, SPEED_BOUNDS, TARGETED_FEATURES

MAX_GENES = 10          # by hand; random genomes have at most genome_factory.MAX_OPERATIONS


def options():
    """What the editor offers, for the page."""
    return {"features": FEATURES, "targeted": TARGETED_FEATURES, "pause_kinds": PAUSE_KINDS,
            "speed_bounds": list(SPEED_BOUNDS), "max_genes": MAX_GENES}


def _number(value, low, high, what):
    if isinstance(value, bool):
        raise ValueError(f"{what} must be a number.")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number.")
    if not low <= value <= high:            # also refuses NaN
        raise ValueError(f"{what} must be from {low:g} to {high:g}.")
    return value


def clean_genes(genes, allow_empty=False):
    """The genes as plain dicts with only the known keys, or ValueError saying what is wrong."""
    if not isinstance(genes, list):
        raise ValueError("The genes must be a list.")
    if not genes and not allow_empty:
        raise ValueError("A genome needs at least one gene.")
    if len(genes) > MAX_GENES:
        raise ValueError(f"A genome has at most {MAX_GENES} genes.")
    out = []
    for number, gene in enumerate(genes, start=1):
        if not isinstance(gene, dict):
            raise ValueError(f"Gene {number} is not a gene.")
        feature = gene.get("feature")
        if feature not in FEATURES:
            raise ValueError(f"Gene {number}: unknown feature {feature!r}.")
        if feature == "speed":
            out.append({"feature": "speed", "rate": round(_number(gene.get("rate"), *SPEED_BOUNDS, f"Gene {number}'s speed"), 2)})
            continue
        clean = {"feature": feature, "position": round(_number(gene.get("position"), 0, 1, f"Gene {number}'s position"), 3)}
        if feature == "pause":
            kind = gene.get("kind", PAUSE_KINDS[0])
            if kind not in PAUSE_KINDS:
                raise ValueError(f"Gene {number}: a pause is {' or '.join(PAUSE_KINDS)}.")
            clean["kind"] = kind
        out.append(clean)
    if sum(g["feature"] == "speed" for g in out) > 1:
        raise ValueError("Only one speed gene: a second one would have no effect.")
    return out


def preview(text, genes):
    """The genes on this line, without rendering: in words, as operations, and as the markup that would be spoken."""
    from backend.speech.tts_formatter import process_tts_string

    if not WORDS.findall(text or ""):
        raise ValueError("The line has no words.")
    genome = Candidate(genes)
    operations = genome.resolve(text)
    plan = process_tts_string(text, operations)
    return {"genes": genes, "described": genome.describe(text), "operations": operations, "markup": plan.text,
            "speed": plan.speed, "instruct": plan.instruct, "warnings": plan.warnings}


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save(name, genes, named_by=None, text=None, based_on=None):
    """A new named genome made by hand. Refused if the name is taken (by any named genome)."""
    name = ratings.clean_name(name)
    genes = clean_genes(genes)
    shown = preview(text, genes) if text else None
    with ratings._lock:
        library = ratings.load_named()
        if any(e["name"].lower() == name.lower() for e in library["genomes"]):
            raise ValueError(f"There is already a genome named {name!r}: choose another name.")
        entry = {"name": name, "named": _now(), "named_by": named_by, "edited": True, "run": None, "genome": None,
                 "genes": genes, "based_on": based_on, "text": text,
                 "described": shown["described"] if shown else None, "markup": shown["markup"] if shown else None,
                 "speed": shown["speed"] if shown else None, "voice": None, "verdict": None, "mos": None}
        library["genomes"].append(entry)
        safe_files.write_json(ratings.NAMED_PATH, library)
    return entry


def remove(name):
    """Delete a genome made by hand (named search genomes are unnamed from their search instead)."""
    with ratings._lock:
        library = ratings.load_named()
        found = [e for e in library["genomes"] if e["name"].lower() == str(name).lower()]
        if not found:
            raise ValueError(f"No genome named {name!r}.")
        if not found[0].get("edited"):
            raise ValueError("That genome was named from a search: take its name off there.")
        library["genomes"] = [e for e in library["genomes"] if e is not found[0]]
        safe_files.write_json(ratings.NAMED_PATH, library)


def library(runs, with_mos=True):
    """
    Every genome that can be picked: the named ones first (newest first), then each search's genomes.
    runs: [(folder, run dict)] the caller may see. with_mos=False leaves out the MOS (listeners rate blind).
    """
    out = []
    for e in reversed(ratings.load_named()["genomes"]):
        out.append({"key": "named:" + e["name"], "source": "edited" if e.get("edited") else "named", "name": e["name"],
                    "run": e.get("run"), "genome": e.get("genome"), "voice": e.get("voice"), "text": e.get("text"),
                    "genes": e["genes"], "described": e.get("described"), "markup": e.get("markup"),
                    "verdict": e.get("verdict"), "named_by": e.get("named_by"), "based_on": e.get("based_on"),
                    "mos": (e.get("mos") or {}).get("mos") if with_mos else None})
    for folder, run in runs:
        rated = ratings.load(folder) if with_mos and (folder / ratings.RATINGS_FILE).is_file() else None
        for g in run["genomes"]:
            score = ratings.mos(rated, ratings.clip_key(g["number"])) if rated else None
            out.append({"key": f"search:{run['id']}/{g['number']}", "source": "search", "name": None,
                        "run": run["id"], "genome": g["number"], "voice": run["voice"], "text": run["text"],
                        "genes": g["genes"], "described": g.get("described"), "markup": g.get("markup"),
                        "verdict": g.get("verdict"), "mos": score["mos"] if score else None})
    return out


def render(reference_audio, reference_text, text, genes, path, seed):
    """Speak the line with these genes (none: the plain line) in the reference's voice. Returns the preview."""
    shown = preview(text, genes)
    from backend.speech.tts_formatter import process_tts_string

    search.render_plan(reference_audio, reference_text, process_tts_string(text, shown["operations"]), Path(path), seed)
    return shown
