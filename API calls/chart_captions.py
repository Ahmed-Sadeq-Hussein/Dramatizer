"""chart_captions.py | Gemma 4 writes the text under each evidence chart (the Graphs tab), for each new set of numbers.

    D:/envs/llm/python.exe chart_captions.py                          # the newest chart set
    D:/envs/llm/python.exe chart_captions.py --folder "D:/Dramatizer 2 renders/evidence charts/all dates"
    D:/envs/llm/python.exe chart_captions.py --again                  # write every text again, even if it is current
    D:/envs/llm/python.exe chart_captions.py --server http://127.0.0.1:11434/v1 --model gemma4   # a model server

`python main_backend.py evidence-charts` runs this by itself after drawing the charts (CHATBOT_PYTHON in .env).

What Gemma gets for each chart (backend/genome/evidence_charts.py writes it into summary.json):
  - the research question, the dates of the scores used, and how many scores, listeners and sanity checks;
  - how the chart is drawn ("note": axes, dashed lines, grey bars, the rules);
  - the exact numbers behind the chart ("numbers"), so it never has to read values off a picture;
  - tools to look further: another chart's numbers, the chart set's totals, and, when an API login is set
    (DRAMATIZER_URL with DRAMATIZER_KEY or DRAMATIZER_USER), the chatbot's API tools (API_REQUEST.TOOLS).

Facts first, in two steps, because a small model mixes up groups when it writes straight from a table:
  1. Gemma names the facts it needs by their path in the numbers, e.g. per_voice[voice=VCTK/p343].best_above_plain
     or chart5:by_features[features=1].genomes. This code looks the values up (resolve()), so a value can never
     be misquoted; a path that is not in the numbers is sent back once with the keys that do exist.
     The chart's headline numbers (headline_facts(): its top-level counts, shares, rules and short lists) are
     always among the facts, so the text answers for the whole chart, not one example group picked from a table.
  2. Gemma writes the text from those facts only, each shown as "path = value", so every number carries the
     group it belongs to. Every number in the text must be one of the facts' values (or a look-up's result); if
     one is not, Gemma is told which and writes once more, and a number still unknown is kept in
     "unverified_numbers" and shown on the page.
The facts are saved with the text, and the page lists them under "What Gemma used".

The texts go into captions.json beside the charts, each under its chart's number with the numbers_id it was
written for. A chart drawn again from the same numbers keeps its text; new numbers (a new cycle of scores) get a
new one the next time this runs. Rule-based sentences ("claim") are not shown to Gemma, so it interprets the
numbers itself rather than repeating ours.
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from chatbot import LocalModel, ServerModel, first_json  # noqa: E402

from backend.genome import evidence_charts as ec  # noqa: E402

MAX_STEPS = 4                 # tool calls per chart, at most
RESULT_CHARS = 6000
MAX_TEXT = 900                # characters of caption text kept

MAX_FACTS = 10
MAX_FACT_CHARS = 400        # a fact longer than this is a whole table: Gemma names single items instead

BACKGROUND = """You write the short text shown under one chart on the Dramatizer's Graphs page.

The Dramatizer clones voices and adds a delivery to a line with a "genome": a few edits (genes) such as a sigh, a
pause, a laugh, stress or a speed change. Listeners scored each genome's clip from 0 (bad) to 5 (excellent).
"Above plain" is a genome's score minus the same listener's score for the plain line (no genome) of that voice.
The sanity check is a machine measurement (words dropped or changed, voice drift, broken pace), not what a
listener heard.
"""

# Step 1: Gemma picks the facts, by their place in the numbers. The values are looked up by code, not by Gemma.
FACTS_INSTRUCTIONS = BACKGROUND + """
Step 1 of 2: choose the facts. You get the research question, the dates of the scores, how the chart is drawn and
the exact numbers behind it (JSON). Choose the facts a researcher needs to answer the question:
- first the facts that ARE the answer: the shares, scores or lists of gene types the question asks about (for
  "how often", the percentages; for "which", the lists), across all the groups, not one example;
- then what they compare with (the value for all genomes, the rule the chart uses), if the numbers have it;
- then how much evidence is behind them (how many genomes or listeners).

Name each fact by its path in the JSON, not by its value:
  a key:                         voices
  a key inside a key:            sanity_check.failed_pct
  one item of a list, by a key:  per_voice[voice=VCTK/p250].best_above_plain
  one item of a list, by place:  per_voice[voice=VCTK/p250].top_genomes[0].listeners
  a whole list or object:        below_every_listener
  another chart's numbers:       chart5:by_features[features=1].genomes
  the chart set's totals:        set:genome_scores   (also set:listeners, set:sanity_checks, set:scores_from, set:scores_to)
The chart's headline numbers are always included (they are listed after the numbers); do not repeat them.
At most """ + str(MAX_FACTS) + """ more facts; none is fine when the headline numbers answer the question.
Reply with exactly one JSON object and nothing else:
  {"facts": ["<path>", "<path>", ...]}
or, to look something up first:  {"tool": "<name>", "arguments": {...}}

Tools:
- chart_numbers: the question and numbers of another chart in this set (to compare).
    number (string, required): "1" to "6"
- chart_set: the totals of this chart set: dates, genome scores, listeners, sanity checks, which charts have data.
"""

# Step 2: Gemma writes from those facts only, each shown with its path, so it can see which group a number belongs to.
WRITE_INSTRUCTIONS = BACKGROUND + """
Step 2 of 2: write the text from the facts you are given. Each fact is "path = value"; the path tells you which group
the value belongs to (a voice, a gene type, a number of features). Write 2 to 4 plain sentences:
- first answer the question directly from the facts that answer it;
- then how it compares with all genomes, if a fact gives that;
- then how much evidence is behind it; a number from one listener is that listener's opinion, and groups with few
  genomes are too few to trust.
The facts marked (headline) are the chart's own summary: the answer starts from them and covers the whole chart;
state each headline list in full (e.g. both what to keep and what to give a lower chance).
A fact about one group (one voice, one gene type) is an example, never the answer on its own.
Use only these facts. Every number you write must be one of the values. Put a number only with the group its path
names: never make a range or a total over groups the facts do not name. Do not state a count (of listeners,
genomes, voices) that is not a fact. Do not write what the chart or data does not show. When a fact is a rule or
a category (e.g. "lower chance", "keep"), use its own words, not stronger ones. Never name a winner or recommend a
genome unless a fact shows the rule for it is met. Write in words: never copy a path or a key name such as
"set:genome_scores" or "lower_chance" into the text. No headings, lists or markdown.
Reply with exactly one JSON object and nothing else: {"answer": "<the text>"}
"""
INSTRUCTIONS = FACTS_INSTRUCTIONS            # what describe() extends with the API tools


# ----------------------------------------------------------------------- tools --
class Tools:
    def __init__(self, summary, api_results=None):
        self.summary, self.api = summary, api_results
        self.names = ["chart_numbers", "chart_set"]
        if api_results is not None:
            from API_REQUEST import TOOL_NAMES
            self.names += list(TOOL_NAMES)

    def describe(self):
        if self.api is None:
            return ""
        from chatbot import tool_list
        return ("\nAPI tools (the project's live data, the same as the 'Ask the Dramatizer' chatbot uses):\n"
                + tool_list())

    def run(self, name, arguments):
        arguments = arguments if isinstance(arguments, dict) else {}
        if name == "chart_numbers":
            number = str(arguments.get("number", "")).strip()
            c = next((c for c in self.summary["charts"] if c["number"] == number), None)
            if c is None:
                return json.dumps({"error": "number is \"1\" to \"6\""})
            if "skipped" in c:
                return json.dumps({"question": c["question"], "no_data": c["skipped"]})
            return json.dumps({"question": c["question"], "numbers": c.get("numbers")}, ensure_ascii=False)
        if name == "chart_set":
            return json.dumps(overview(self.summary), ensure_ascii=False)
        if self.api is not None and name in self.names:
            return self.api.run_tool(name, arguments)
        return json.dumps({"error": f"No tool {name!r}. The tools are: {', '.join(self.names)}."})


def overview(summary):
    r = summary.get("range") or {}
    return {"scores_from": r.get("from") or "the first score", "scores_to": r.get("to") or "now",
            "genome_scores": summary.get("genome_scores"), "listeners": summary.get("listeners"),
            "sanity_checks": summary.get("sanity_checks"), "sanity_source": summary.get("sanity_source"),
            "charts": [{"number": c["number"], "question": c["question"],
                        **({"no_data": c["skipped"]} if "skipped" in c else {})} for c in summary["charts"]]}


# ------------------------------------------------------------------ checking --
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def numbers_in(text):
    return [float(x) for x in NUMBER.findall(str(text))]


def unknown_numbers(text, context):
    """Numbers in the text that appear nowhere in what Gemma was given (small counts 0..6 are always allowed)."""
    known = set(numbers_in(context))
    known |= {abs(x) for x in known}
    return sorted({x for x in numbers_in(text) if abs(x) not in known and not (x == int(x) and 0 <= x <= 6)})


# --------------------------------------------------------------------- writer --
def chart_message(summary, c):
    o = overview(summary)
    return (f"Question {c['number']}: {c['question']}\n"
            f"Scores used: given from {o['scores_from']} to {o['scores_to']}: {o['genome_scores']} genome scores from "
            f"{o['listeners']} listeners, {o['sanity_checks']} sanity checks ({o['sanity_source']}).\n"
            f"How the chart is drawn: {c['note']}\n"
            f"The numbers behind the chart (JSON):\n{json.dumps(c['numbers'], ensure_ascii=False)}")


# ------------------------------------------------------------- fact paths --
STEP = re.compile(r"([^.\[\]]+)|\[([^\]]*)\]")


class PathError(ValueError):
    pass


def resolve(path, summary, c):
    """The value at a fact path (see FACTS_INSTRUCTIONS), looked up in the numbers; PathError if there is none."""
    path = str(path).strip()
    if path.startswith("set:"):
        root, path = overview(summary), path[4:]
    elif re.match(r"chart\s*\d+\s*:", path):
        number, path = re.match(r"chart\s*(\d+)\s*:(.*)", path).groups()
        other = next((x for x in summary["charts"] if x["number"] == number), None)
        if other is None or "numbers" not in other:
            raise PathError(f"chart {number} has no numbers" + (f" ({other['skipped']})" if other else ""))
        root = other["numbers"]
    else:
        root = c["numbers"]
    value = root
    for key, selector in STEP.findall(path.strip(".")):
        if key:
            if not isinstance(value, dict) or key.strip() not in value:
                keys = ", ".join(value) if isinstance(value, dict) else "none (not an object)"
                raise PathError(f"no key {key.strip()!r} here; the keys are: {keys}")
            value = value[key.strip()]
        elif "=" in selector:
            k, want = (s.strip().strip("'\"") for s in selector.split("=", 1))
            if not isinstance(value, list):
                raise PathError(f"[{selector}] needs a list")
            match = [x for x in value if isinstance(x, dict) and str(x.get(k)) == want]
            if not match:
                have = sorted({str(x.get(k)) for x in value if isinstance(x, dict)})[:30]
                raise PathError(f"no item with {k}={want}; there are: {', '.join(have)}")
            value = match[0]
        else:
            try:
                value = value[int(selector)]
            except (ValueError, IndexError, TypeError, KeyError):
                raise PathError(f"[{selector}] is not a place in a list here") from None
    return value


def fact_line(path, value, headline=False):
    return f"- {path} = {json.dumps(value, ensure_ascii=False)}" + (" (headline)" if headline else "")


def _plain(v):
    return v is None or isinstance(v, (bool, int, float, str))


def headline_terms(value):
    """What would show a headline fact is in a text: its numbers and its words (a list's items)."""
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [value]
    if isinstance(value, str):
        return [value] if len(value) <= 40 else []
    items = value.values() if isinstance(value, dict) else value
    return [t for x in items for t in headline_terms(x)]


def misses_headline(text, headline):
    """True when the text uses none of the headline facts' numbers or words (it answers from an example only)."""
    terms = [t for _, v in headline for t in headline_terms(v)
             if not (isinstance(t, (int, float)) and t == int(t) and 0 <= t <= 6)]   # small counts match by chance
    if not terms:
        return False
    numbers = set(numbers_in(text))
    low = text.lower()
    return not any((t in numbers) if isinstance(t, (int, float)) else (t.lower() in low) for t in terms)


def headline_facts(c):
    """The chart's top-level numbers that sum it up: [(path, value)] for each value that is a number or a word,
    a short list of them (at most 8) or a small object of them (at most 4 keys). Long tables are left to Gemma."""
    out = []
    for key, value in (c.get("numbers") or {}).items():
        if (_plain(value)
                or (isinstance(value, list) and len(value) <= 8 and all(_plain(x) for x in value))
                or (isinstance(value, dict) and len(value) <= 4 and all(_plain(x) for x in value.values()))):
            out.append((key, value))
    return out


def pick_facts(model, tools, summary, c, messages, used, context):
    """Step 1: Gemma names the facts (paths), code looks up their values. Returns [(path, value)]."""
    asked_again = False
    facts = headline_facts(c)
    headline = len(facts)
    for _ in range(MAX_STEPS + 2):
        reply = model.chat(messages)
        call = first_json(reply)
        if call and "tool" in call and len(used) < MAX_STEPS:
            name, arguments = str(call["tool"]), call.get("arguments") or {}
            result = tools.run(name, arguments)
            used.append(name)
            context.append(result)
            shown = result if len(result) <= RESULT_CHARS else result[:RESULT_CHARS] + " ... (cut)"
            messages += [{"role": "assistant", "content": json.dumps({"tool": name, "arguments": arguments})},
                         {"role": "user", "content": f"Result of {name}:\n{shown}\n\nLook up more, or reply "
                                                     '{"facts": [...]} with the paths of the facts.'}]
            continue
        paths = call.get("facts") if call and isinstance(call.get("facts"), list) else []
        bad = []
        for p in paths[:MAX_FACTS + headline]:
            if any(p == q for q, _ in facts):
                continue
            try:
                value = resolve(p, summary, c)
            except PathError as error:
                bad.append(f"{p}: {error}")
                continue
            if len(json.dumps(value, ensure_ascii=False)) > MAX_FACT_CHARS:
                bad.append(f"{p}: a whole table, too long for one fact; name single items in it instead")
                continue
            facts.append((str(p), value))
        if (bad or not (paths or headline)) and not asked_again:
            asked_again = True
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": ("These paths cannot be used:\n" + "\n".join(bad)
                                                      if bad else "Reply with the paths of the facts.")
                                                     + '\nReply {"facts": [...]} with the paths to add.'}]
            continue
        break
    if not facts:
        raise RuntimeError(f"Gemma named no fact that is in the numbers for chart {c['number']}")
    return facts


def write_one(model, tools, summary, c):
    """Gemma's text for one chart: {"text", "facts": [{"path", "value"}], "tools", "unverified_numbers"?}."""
    first = chart_message(summary, c)
    headline = headline_facts(c)
    if headline:
        first += ("\n\nAlways included, the chart's headline numbers:\n"
                  + "\n".join(fact_line(p, v) for p, v in headline))
    messages = [{"role": "system", "content": FACTS_INSTRUCTIONS + tools.describe()}, {"role": "user", "content": first}]
    used, looked_up = [], []
    facts = pick_facts(model, tools, summary, c, messages, used, looked_up)
    o = overview(summary)
    marked = {p for p, _ in headline}
    facts_text = "\n".join(fact_line(p, v, p in marked) for p, v in facts)
    allowed = facts_text + "\n" + "\n".join(looked_up)          # numbers the text may use: the facts and look-ups
    writing = [{"role": "system", "content": WRITE_INSTRUCTIONS},
               {"role": "user", "content": f"Question {c['number']}: {c['question']}\n"
                                           f"Scores given from {o['scores_from']} to {o['scores_to']}.\n"
                                           f"How the chart is drawn: {c['note']}\n"
                                           f"The facts (checked against the data):\n{facts_text}"}]
    sent_back = False                    # numbers not among the facts are sent back once
    headline_asked = False               # a text that uses no headline fact is sent back once
    last = 3
    for attempt in range(last + 1):
        reply = model.chat(writing)
        call = first_json(reply)
        if not (call and isinstance(call.get("answer"), str)):
            if attempt < last:                   # cut off or not JSON: ask again, shorter
                writing += [{"role": "assistant", "content": reply[:400]},
                            {"role": "user", "content": 'That was not one whole JSON object. Reply with only '
                                                        '{"answer": "<2 to 4 sentences>"}, shorter.'}]
                continue
            raise RuntimeError(f"Gemma gave no text for chart {c['number']}: {reply[:200]!r}")
        text = re.sub(r"\s+", " ", call["answer"]).strip()[:MAX_TEXT]
        if not text:
            raise RuntimeError(f"Gemma gave an empty text for chart {c['number']}")
        unknown = unknown_numbers(text, allowed + f"\n{o['scores_from']} {o['scores_to']}")
        if not unknown and not headline_asked and attempt < last and misses_headline(text, headline):
            headline_asked = True
            writing += [{"role": "assistant", "content": json.dumps({"answer": text})},
                        {"role": "user", "content": "This answers from one example only. Start from the (headline) "
                                                    "facts, which cover the whole chart:\n"
                                                    + "\n".join(fact_line(p, v) for p, v in headline)
                                                    + '\nWrite the text again. Reply {"answer": "..."}.'}]
            continue
        if not unknown or sent_back or attempt == last:
            out = {"text": text, "facts": [{"path": p, "value": v} for p, v in facts], "tools": used}
            if unknown:
                out["unverified_numbers"] = unknown
            return out
        sent_back = True
        writing += [{"role": "assistant", "content": json.dumps({"answer": text})},
                    {"role": "user", "content": "These numbers are not among the facts: "
                                                + ", ".join(f"{x:g}" for x in unknown)
                                                + '. Write the text again using only the facts. Reply {"answer": "..."}.'}]


def latest_folder():
    best = None
    for path in Path(ec.CHARTS_DIR).glob("*/summary.json"):
        try:
            made = json.loads(path.read_text(encoding="utf-8")).get("made")
        except (OSError, ValueError):
            continue
        if made and (best is None or made > best[0]):
            best = (made, path.parent)
    if best is None:
        raise SystemExit("No charts yet: run python main_backend.py evidence-charts first.")
    return best[1]


def connect_api():
    """The chatbot's API tools when a login is set in the environment, else None (the chart's own tools still work)."""
    if not (os.environ.get("DRAMATIZER_KEY") or os.environ.get("DRAMATIZER_USER")):
        return None
    try:
        from API_REQUEST import Results
        return Results.connect()
    except Exception as error:                     # no API running: Gemma still has the chart's numbers
        print(f"API tools not available ({type(error).__name__}: {error}); using the charts' own numbers only.")
        return None


def write_captions(folder, model, again=False, api_results=None, log=print):
    folder = Path(folder)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    saved = ec.read_captions(folder)
    tools = Tools(summary, api_results)
    name = getattr(model, "name", "gemma")
    written = 0
    for c in summary["charts"]:
        if "skipped" in c or "numbers" not in c:
            continue
        old = saved.get(c["number"])
        if old and old.get("numbers_id") == c["numbers_id"] and not again:
            log(f"  {c['number']}. current, kept")
            continue
        started = time.time()
        out = write_one(model, tools, summary, c)
        saved[c["number"]] = {"numbers_id": c["numbers_id"], "question": c["question"], "model": name,
                              "made": datetime.now().strftime("%Y-%m-%d %H:%M"), **out}
        written += 1
        log(f"  {c['number']}. written in {time.time() - started:.0f} s"
            + (f" (unverified numbers: {out['unverified_numbers']})" if out.get("unverified_numbers") else ""))
        ec.safe_files.write_json(folder / ec.CAPTIONS, saved)        # after each chart, so a stop loses little
    ec.attach_captions(summary, folder)
    ec.write_comments(summary, folder)
    ec.safe_files.write_json(folder / "summary.json", summary)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description="Gemma 4 writes the text under each evidence chart.")
    parser.add_argument("--folder", help="a chart set's folder (default: the newest one)")
    parser.add_argument("--again", action="store_true", help="write every text again, even when it is current")
    parser.add_argument("--model", default=os.environ.get("CHATBOT_MODEL") or "gemma-4-E4B-it",
                        help="a folder in LLM_LIBRARY_DIR, or the server's model name")
    parser.add_argument("--server", help="an OpenAI-compatible address instead of a local model")
    args = parser.parse_args(argv)
    folder = Path(args.folder) if args.folder else latest_folder()
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    todo = [c for c in summary["charts"] if "numbers" in c]
    if not todo:
        print(f"{folder.name}: no chart with data, nothing to write.")
        return 0
    print(f"Charts: {folder}")
    model = ServerModel(args.server, args.model) if args.server else LocalModel(args.model)
    api = connect_api()
    try:
        n = write_captions(folder, model, args.again, api)
    finally:
        if api is not None:
            api.close()
    print(f"Gemma wrote {n} text(s). They are in {folder / ec.CAPTIONS} and comments.txt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
