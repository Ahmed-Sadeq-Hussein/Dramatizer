"""test_emphasis_llms.py | Which small LLM picks the right words to stress, and how fast?

Runs every model on emphasis_testset.json: 24 lines, 11 of them the same
sentence with different context, so the right word to stress must move with
the context. The expected values for every word were written by Claude Opus
5.5 before any model ran (the benchmark), together with its predicted timings.

For each model it measures:

    load_s           seconds from disk to ready (4-bit)
    vram_gb          GPU memory after loading
    first_call_s     the first answer (includes warm-up)
    per_line_s       seconds per line afterwards (mean), and tokens per second
    valid_format     share of answers that came back as one value per word, in order
    focus_accuracy   share of lines where the highest-valued word is a focus word
                     (a tie of more than two top words counts as wrong)
    groups_correct   context groups where EVERY variant was right (car, say, tom)
    mae              mean distance from the benchmark values, per word
    rank_corr        Spearman correlation with the benchmark values, per line, averaged
    over_stressed    words at 0.7 or more, minus the benchmark's count, per line
    agreement        with 3 sampled answers: share of lines where all 3 pick the same top word
    spread           mean standard deviation of a word's value over the 3 samples

    python -m backend.emphasis.test_emphasis_llms                    # all models in LLM_LIBRARY_DIR
    python -m backend.emphasis.test_emphasis_llms Qwen3-4B           # one model
    python -m backend.emphasis.test_emphasis_llms Qwen3-4B:think     # with its thinking mode on
    python -m backend.emphasis.test_emphasis_llms --samples 0 ...    # skip the 3 sampled answers (agreement, spread)
    python -m backend.emphasis.test_emphasis_llms --fresh ...        # forget saved progress and start the models over

Every line is saved as soon as it is answered (output/emphasis_test/progress/),
so a run can be stopped at any time (Ctrl+C, closing the window) and started
again with the same command: it carries on from the next unanswered line.
results.json and report.md are rewritten every time a model finishes.
"""
import json
import os
import statistics
import sys
import time

import numpy as np

from backend import model
from backend.emphasis.emphasis import _normal, ask, parse_values

TESTSET = model.DATA_DIR / "emphasis_testset.json"
OUT_DIR = model.PROJECT_DIR / "output" / "emphasis_test"
PROGRESS_DIR = OUT_DIR / "progress"
# "name:think" runs the same model with its thinking mode on
MODELS = ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-4B:think", "gemma-4-E2B-it", "gemma-4-E4B-it"]
SAMPLES = 3     # sampled answers per line for agreement and spread; --samples 0 skips them


def ranks(values):
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable").astype(float)
    # Average ranks of ties
    for v in set(values):
        idx = [i for i, x in enumerate(values) if x == v]
        order[idx] = order[idx].mean()
    return order


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def top_correct(words, values, focus):
    top = max(values)
    best = {_normal(w) for w, v in zip(words, values) if v == top}
    return len(best) <= 2 and bool(best & {_normal(f) for f in focus})


def _save_json(path, data):
    """Write through a temporary file, so stopping mid-write never leaves a broken file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _progress_file(spec):
    return PROGRESS_DIR / f"{spec.replace(':', '_')}.json"


def test_model(spec, items):
    """Answer every line not yet in this model's progress file, saving after each one, then summarise."""
    import torch

    name, _, mode = spec.partition(":")
    think = mode == "think"

    progress_file = _progress_file(spec)
    progress = json.loads(progress_file.read_text(encoding="utf-8")) if progress_file.exists() else {}
    # Saved lines only count if they were made the same way (same number of samples)
    if progress.get("samples") != SAMPLES:
        progress = {"samples": SAMPLES}
    rows = {row["id"]: row for row in progress.get("lines", [])}
    todo = [item for item in items if item["id"] not in rows]
    if rows:
        print(f"  resuming: {len(rows)} of {len(items)} lines already saved", flush=True)

    if todo:
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model.emphasis_model(name)
        load_s = time.perf_counter() - started
        # The first load's timings are the ones reported
        progress.setdefault("load_s", round(load_s, 1))
        progress.setdefault("vram_gb", round(torch.cuda.memory_allocated() / 1e9, 2))

        started = time.perf_counter()
        ask("This is a short warm-up line.", None, name, think=think)
        progress.setdefault("first_call_s", round(time.perf_counter() - started, 2))

        for item in todo:
            started = time.perf_counter()
            answer, tokens = ask(item["line"], item["context"], name, think=think)
            seconds = time.perf_counter() - started
            pairs, ok = parse_values(answer, item["line"])
            values = [v for _, v in pairs]
            words = item["line"].split()

            sampled = [[v for _, v in parse_values(ask(item["line"], item["context"], name, sample=True, think=think)[0], item["line"])[0]]
                       for _ in range(SAMPLES)]
            tops = [max(range(len(s)), key=s.__getitem__) for s in sampled]
            consistency = {"agree": len(set(tops)) == 1, "spread": float(np.mean(np.std(np.array(sampled), axis=0)))} \
                if SAMPLES > 1 else {"agree": None, "spread": None}

            rows[item["id"]] = {
                "id": item["id"], "group": item["group"], "answer": answer, "values": values, "valid": ok,
                "seconds": round(seconds, 3), "tokens": tokens,
                "correct": top_correct(words, values, item["focus"]),
                "mae": float(np.mean(np.abs(np.array(values) - np.array(item["gold"])))),
                "rank_corr": spearman(values, item["gold"]),
                "over_stressed": sum(v >= 0.7 for v in values) - sum(g >= 0.7 for g in item["gold"]),
                **consistency,
            }
            progress["peak_vram_gb"] = round(max(progress.get("peak_vram_gb", 0), torch.cuda.max_memory_allocated() / 1e9), 2)
            progress["lines"] = [rows[i["id"]] for i in items if i["id"] in rows]
            _save_json(progress_file, progress)
            print(f"  {item['id']:15s} {'ok ' if ok else 'BAD'} {'right' if rows[item['id']]['correct'] else 'wrong'}  "
                  f"{seconds:6.2f} s  {tokens} tokens  ({len(rows)}/{len(items)} saved)", flush=True)

        model.unload(f"emphasis:{name}")

    return summarise(progress, [rows[item["id"]] for item in items])


def summarise(progress, rows):
    groups = {}
    for row in rows:
        if row["group"]:
            groups.setdefault(row["group"], []).append(row["correct"])
    sampled = progress["samples"] > 1

    return {
        "load_s": progress["load_s"],
        "vram_gb": progress["vram_gb"],
        "peak_vram_gb": progress["peak_vram_gb"],
        "first_call_s": progress["first_call_s"],
        "per_line_s": round(statistics.mean(r["seconds"] for r in rows), 2),
        "tokens_per_s": round(sum(r["tokens"] for r in rows) / sum(r["seconds"] for r in rows), 1),
        "valid_format": round(statistics.mean(r["valid"] for r in rows), 3),
        "focus_accuracy": round(statistics.mean(r["correct"] for r in rows), 3),
        "groups_correct": f"{sum(all(g) for g in groups.values())}/{len(groups)}",
        "mae": round(statistics.mean(r["mae"] for r in rows), 3),
        "rank_corr": round(statistics.mean(r["rank_corr"] for r in rows), 3),
        "over_stressed": round(statistics.mean(r["over_stressed"] for r in rows), 2),
        "agreement": round(statistics.mean(r["agree"] for r in rows), 3) if sampled else "not run",
        "spread": round(statistics.mean(r["spread"] for r in rows), 3) if sampled else "not run",
        "lines": rows,
    }


def save_results(new, expected):
    """Merge these models into results.json (other models' earlier results stay) and rewrite report.md."""
    path = OUT_DIR / "results.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    results.update(new)
    results = {n: results[n] for n in MODELS if n in results} | {n: r for n, r in results.items() if n not in MODELS}
    _save_json(path, results)
    (OUT_DIR / "report.md").write_text(report(results, expected), encoding="utf-8")


def report(results, expected):
    names = list(results)
    metrics = [("load_s", "Load (s)"), ("vram_gb", "GPU memory (GB)"), ("first_call_s", "First answer (s)"),
               ("per_line_s", "Per line (s)"), ("tokens_per_s", "Tokens / s"), ("valid_format", "Valid format"),
               ("focus_accuracy", "Focus accuracy"), ("groups_correct", "Context groups all right"),
               ("mae", "Distance from benchmark"), ("rank_corr", "Rank correlation"),
               ("over_stressed", "Extra stressed words / line"), ("agreement", "Same top word, 3 samples"),
               ("spread", "Value spread, 3 samples")]
    lines = ["# Emphasis LLM test", "",
             "Benchmark: Claude Opus 5.5's values (focus accuracy 1.0, distance 0, rank correlation 1 by definition).",
             "Expected = Opus 5.5's prediction written before running.", "",
             "| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key, label in metrics:
        cells = []
        for n in names:
            value = results[n][key]
            guess = expected.get(n, {}).get(key)
            cells.append(f"{value} (expected {guess})" if guess is not None else f"{value}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += ["", "## Per line: top word chosen", "", "| Line | Benchmark focus | " + " | ".join(names) + " |",
              "|---|---|" + "---|" * len(names)]
    items = {i["id"]: i for i in json.loads(TESTSET.read_text(encoding="utf-8"))["items"]}
    for k, item in enumerate(items.values()):
        words = item["line"].split()
        cells = []
        for n in names:
            row = results[n]["lines"][k]
            top = words[int(np.argmax(row["values"]))]
            cells.append(f"{'✓' if row['correct'] else '✗'} {top}")
        lines.append(f"| {item['id']} | {', '.join(item['focus'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    data = json.loads(TESTSET.read_text(encoding="utf-8"))
    items = data["items"]
    for item in items:
        assert len(item["line"].split()) == len(item["gold"]), item["id"]

    global SAMPLES
    args = sys.argv[1:]
    if "--samples" in args:
        at = args.index("--samples")
        SAMPLES = int(args[at + 1])
        del args[at:at + 2]
    fresh = "--fresh" in args
    args = [a for a in args if a != "--fresh"]
    names = args or MODELS

    for name in names:
        print(f"\n{name}", flush=True)
        if fresh:
            _progress_file(name).unlink(missing_ok=True)
        result = test_model(name, items)
        print(f"  {({k: v for k, v in result.items() if k != 'lines'})}", flush=True)
        save_results({name: result}, data["expected"])
        print(f"  saved to {OUT_DIR / 'results.json'}", flush=True)

    print(f"\nWrote {OUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
