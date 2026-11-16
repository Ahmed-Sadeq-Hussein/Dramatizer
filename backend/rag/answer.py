"""answer.py | The last steps of the evidence milestone: an answer that cites its records, or says there is none.

    python main_backend.py rag answer "Which genomes of voice VCTK/p250 scored better than its plain line?"
    python main_backend.py rag answer "..." --as USERNAME   # only what that account may see
    python main_backend.py rag answer --check               # the 10 questions and the paraphrases, checked

A question goes one of three ways (route):

    summary   one of the genome questions that need counting over many records (best per voice, artifacts,
              bad MOS, better than the plain line, by number of features, exclude, consistently good).
              Step 05 showed that no retriever finds these in its top 10 (q02, q03, q06 have hundreds of
              relevant records), so they are counted from the records' fields, by the written rules of
              labels.py, and limited to one voice when the question names one.
    genome    "genome N of voice V": that genome's clip records, its plain line and its run, looked up.
    search    anything else: BM25 (the step 03 baseline, which step 05 found best). If its best record
              scores below CUTOFF, nothing in the evidence answers the question, and the answer says so
              instead of quoting the nearest record (safe absence).

Every answer lists the records it used (citations: ID and revision), so it can be checked against the
corpus, and it is made only from the records the asker may see: an account that may not see other
listeners' scores (a listener or a researcher) gets their own clip scores and the sanity checks, no runs and no one
else's scores, the same rule as the API (backend/api/server.py, sees_others).

CUTOFF. In the 10 questions, the two that nothing answers score 0.001 (q09) and 1.6 (q10) with BM25;
the answerable lookups score 21.2 (q07) and 24.5 (q08). CUTOFF = 6 sits between them (about their
geometric middle). With two unanswerable and two answerable search questions it is a working choice,
not a measured optimum; --check shows the margin on both sides and the paraphrases test it a little more.

The summary rules are the same ones that wrote the answer key (labels.py), so checking a summary answer
against the key only shows that routing picked the right rule. That is what the paraphrases check:
questions worded differently from questions.json, with the route each should take.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from backend import safe_files
from backend.rag import baseline, labels
from backend.rag import corpus as evidence

CUTOFF = 6.0
CITE = 8                      # citations listed in full; the rest are counted
PARAPHRASES_PATH = Path(__file__).resolve().parent / "paraphrases.json"
GOOD = 4.0                    # backend/genome/report.py: a score this high is good
GENOME = re.compile(r"\bgenome\s*(?:number\s*)?#?(\d{1,2})\b", re.IGNORECASE)


# ------------------------------------------------------------------ who may see what --
def visible(records, listener=None):
    """The records an asker may see. listener=None: everything (an admin, a researcher, this PC).
    A listener's display name: their own clip scores and the sanity checks only."""
    if listener is None:
        return list(records)
    return [r for r in records if r["kind"] == "sanity" or r["kind"] == "clip" and r["fields"]["listener"] == listener]


# ------------------------------------------------------------------ the question --
def voice_of(question):
    m = baseline.VOICE.search(question)
    if not m:
        return None
    names = {"crema-d": "CREMA-D", "libritts-r": "LibriTTS-R", "vctk": "VCTK"}
    return f"{names[m.group(1).lower()]}/{m.group(2)}"


INTENTS = [   # (name, test on the lower-cased question), first match wins
    ("by_features", lambda q: re.search(r"\b(number|how many|count)\b.{0,40}\b(features?|genes?)\b", q)
                              or re.search(r"\b(features?|genes?)\b.{0,20}\b(count|number)\b", q)),
    ("artifacts", lambda q: re.search(r"artifact|artefact|glitch|distort|robotic|crackl", q)),
    ("exclude", lambda q: re.search(r"\b(exclude|excluded|drop|discard|leave out|rule out|remove|avoid)\b", q)),
    ("consistent", lambda q: re.search(r"\bconsistent(ly)?\b|\breliabl[ey]\b|\bacross (all |the )?voices\b", q)),
    ("bad", lambda q: re.search(r"\b(bad|worst|poor(ly)?|low(est)?)\b.{0,30}\b(mos|scores?|ratings?|rated)\b"
                                r"|\b(mos|scores?|ratings?|rated)\b.{0,20}\b(bad|badly|worst|poor(ly)?|low(est)?)\b"
                                r"|\bworse than\b", q)),
    ("better", lambda q: re.search(r"\b(better|beat|beats|above|outperform\w*|improve\w*)\b.{0,40}"
                                   r"\b(plain|baseline|unedited|original)\b", q)),
    ("best", lambda q: re.search(r"\b(best|top|highest)\b", q)),
]


def route(question):
    """("summary", intent, voice) | ("genome", number, voice) | ("search", None, None)."""
    q = question.lower()
    voice = voice_of(question)
    for name, test in INTENTS:
        if test(q):
            return "summary", name, voice
    g = GENOME.search(question)
    if g and voice:
        return "genome", int(g.group(1)), voice
    return "search", None, None


# ------------------------------------------------------------------ the answers --
def _clips(records, voice=None):
    return [r for r in records if r["kind"] == "clip" and r["fields"]["genome"] is not None
            and (voice is None or r["fields"]["voice"] == voice)]


def _who(r):
    f = r["fields"]
    return f"{f['voice']} genome {f['genome']} ({f['listener']}: {f['score']:g}" + (
        f", {f['above_plain']:+g} vs plain)" if f["above_plain"] is not None else ")")


def summary(intent, records, voice=None):
    """(answer text, {record id: grade}) for one of the summary questions, from the records' fields."""
    clips = _clips(records, voice)
    scope = f" for {voice}" if voice else ""
    if not clips and intent != "artifacts":
        return None, {}
    if intent == "best":
        best = defaultdict(list)
        for c in clips:
            best[(c["fields"]["voice"], c["fields"]["listener"])].append(c)
        lines, cited = [], {}
        for (v, who), cs in sorted(best.items()):
            top = max(c["fields"]["score"] for c in cs)
            winners = sorted(c["fields"]["genome"] for c in cs if c["fields"]["score"] == top)
            lines.append(f"{v} ({who}): genome {', '.join(map(str, winners))} with {top:g}"
                         + (" (a tie)" if len(winners) > 1 else ""))
            for c in cs:
                if c["fields"]["score"] == top:
                    cited[c["id"]] = 2
        ties = sum("(a tie)" in x for x in lines)
        heard_by = Counter(len({c["fields"]["listener"] for c in clips
                                if (c["fields"]["voice"], c["fields"]["genome"]) == key})
                           for key in {(c["fields"]["voice"], c["fields"]["genome"]) for c in clips})
        alone = heard_by.get(1, 0)
        caution = (f"\nNot a winner: {alone} of {sum(heard_by.values())} genomes{scope} were scored by one listener "
                   "only, so each line is one listener's highest score, not a result. A winner needs at least 2 "
                   "listeners (GET /evidence/snapshot gives the rule and each voice's status)." if alone else "")
        return (f"The highest-scored genome{scope}, per voice and listener ({len(lines)}; {ties} have a tie):\n  "
                + "\n  ".join(lines) + caution), cited
    if intent == "artifacts":
        heard = [c for c in clips if c["fields"]["artifact"] is True]
        failed = [r for r in records if r["kind"] == "sanity" and r["fields"]["verdict"] == "failed"
                  and (voice is None or r["fields"]["voice"].replace("_", "/") == voice)]
        types = Counter(t for r in failed for t in r["fields"]["gene_types"])
        text = (f"Listeners marked an artifact in {len(heard)} genome clip(s){scope}"
                + (": " + "; ".join(_who(c) for c in heard[:CITE]) if heard else "") + ". "
                f"The sanity check failed {len(failed)} genome render(s){scope}"
                + (", most often with " + ", ".join(f"{t} ({n})" for t, n in types.most_common(4)) if failed else "")
                + ". A failed sanity check is the closest evidence of an artifact where no listener answered.")
        return text, {**{c["id"]: 2 for c in heard}, **{r["id"]: 1 for r in failed}}
    if intent == "bad":
        bad = sorted((c for c in clips if c["fields"]["above_plain"] is not None and c["fields"]["above_plain"] < 0),
                     key=lambda c: c["fields"]["above_plain"])
        rated = [c for c in clips if c["fields"]["above_plain"] is not None]
        share = {}
        for t in {t for c in rated for t in c["fields"]["gene_types"]}:
            having = [c for c in rated if t in c["fields"]["gene_types"]]
            if len(having) >= 5:
                share[t] = (sum(c["fields"]["above_plain"] < 0 for c in having) / len(having), len(having))
        worst = sorted(share.items(), key=lambda x: -x[1][0])[:4]
        text = (f"{len(bad)} of {len(rated)} rated genomes{scope} scored below their plain line"
                + (": worst " + "; ".join(_who(c) for c in bad[:5]) if bad else "") + "."
                + (" The gene types most often below the plain line: "
                   + ", ".join(f"{t} ({s:.0%} of {n})" for t, (s, n) in worst) + "." if worst and not voice else ""))
        return text, {c["id"]: 2 for c in bad}
    if intent == "better":
        good = sorted((c for c in clips if c["fields"]["above_plain"] is not None and c["fields"]["above_plain"] > 0),
                      key=lambda c: -c["fields"]["above_plain"])
        rated = [c for c in clips if c["fields"]["above_plain"] is not None]
        listed = good if voice else good[:CITE]
        text = (f"{len(good)} of {len(rated)} rated genomes{scope} scored better than their plain line"
                + (": " + "; ".join(_who(c) for c in listed) if listed else "") + "."
                + (" Each genome was heard by one listener, so a difference of 0.5 is one step of the scale." if good else ""))
        runs = {r["id"]: 1 for r in records if r["kind"] == "run" and r["fields"]["better_than_plain"] > 0
                and (voice is None or r["fields"]["voice"] == voice)}
        return text, {**runs, **{c["id"]: 2 for c in good}}
    if intent == "by_features":
        groups = defaultdict(list)
        for c in clips:
            if c["fields"]["above_plain"] is not None:
                groups[c["fields"]["features"]].append(c["fields"]["above_plain"] > 0)
        lines = [f"{k} feature(s): {sum(v)} of {len(v)} ({sum(v) / len(v):.0%})" for k, v in sorted(groups.items())]
        return f"Genomes{scope} better than their plain line, by number of active features:\n  " + "\n  ".join(lines), \
            {c["id"]: 1 for c in clips}
    if intent == "exclude":
        cited = labels.exclude(records)
        cited = {i: g for i, g in cited.items() if voice is None or i.split(":", 1)[1].replace("_", "/").find(voice) >= 0}
        worse = [c for c in clips if c["id"] in cited and cited[c["id"]] == 2]
        text = (f"The gene types the evidence gives a lower chance are {', '.join(sorted(labels.LOWER_CHANCE))} "
                f"(evidence chart 6): {len(worse)} rated genome(s){scope} with them scored below their plain line"
                + (", e.g. " + "; ".join(_who(c) for c in worse[:4]) if worse else "") + ". "
                "Acting on this, hissing and vocalized noise are now disabled in the genome factory and speed is "
                "limited to 0.85-1.15 (backend/genome/genome_factory.py).")
        return text, cited
    if intent == "consistent":
        groups = defaultdict(list)
        for c in clips:
            groups[tuple(sorted(c["fields"]["gene_types"]))].append(c)
        steady = []
        for types, cs in groups.items():
            good_voices = {c["fields"]["voice"] for c in cs if c["fields"]["score"] >= GOOD}
            if types and len(good_voices) >= 2:
                steady.append((len(good_voices), types, cs))
        steady.sort(key=lambda x: (-x[0], x[1]))
        if not steady:
            return (f"No combination of gene types{scope} scored {GOOD:g} or more on two or more voices: each "
                    "random genome is a different combination, so none was heard often enough."), \
                {c["id"]: 1 for c in clips if c["fields"]["score"] >= GOOD}
        lines = [f"{' + '.join(t)}: good on {n} voices, mean {mean(c['fields']['score'] for c in cs):.2f}"
                 for n, t, cs in steady[:CITE]]
        return (f"Gene-type combinations{scope} scored {GOOD:g}+ on at least two voices:\n  " + "\n  ".join(lines)), \
            {c["id"]: 2 for _, _, cs in steady for c in cs if c["fields"]["score"] >= GOOD}
    raise ValueError(intent)


def one_genome(number, voice, records):
    mine = [r for r in records if r["kind"] == "clip" and r["fields"]["voice"] == voice and r["fields"]["genome"] == number]
    if not mine:
        return None, {}
    plain = [r for r in records if r["kind"] == "clip" and r["fields"]["voice"] == voice and r["fields"]["genome"] is None]
    runs = [r for r in records if r["kind"] == "run" and r["fields"]["voice"] == voice]
    return " ".join(r["text"] for r in mine), {**{r["id"]: 2 for r in mine}, **{r["id"]: 1 for r in plain + runs}}


def answer(question, records, index=None, listener=None, cutoff=CUTOFF):
    """{"question", "route", "absent", "answer", "citations": [{"id", "revision", "grade"}], "cited", "top_score"}"""
    seen = visible(records, listener)
    way, what, voice = route(question)
    text, cited, top = None, {}, None
    if way == "summary":
        text, cited = summary(what, seen, voice)
    elif way == "genome":
        text, cited = one_genome(what, voice, seen)
    else:
        index = index if index is not None and listener is None else baseline.BM25(seen)
        hits = index.search(question, k=3)
        top = hits[0]["score"] if hits else 0.0
        if top >= cutoff:
            byid = {r["id"]: r for r in seen}
            strong = [h for h in hits if h["score"] >= cutoff]
            text = "\n".join(f"[{h['id']}] {byid[h['id']]['text']}" for h in strong)
            cited = {h["id"]: 2 for h in strong}
    revision = {r["id"]: r["revision"] for r in seen}
    order = sorted(cited, key=lambda i: (-cited[i], i))
    absent = text is None
    if absent:
        text = ("No record in the evidence answers this"
                + (f" (the closest scores {top:.1f}, below the cut-off {cutoff:g})" if top is not None else
                   f" ({what if way == 'genome' else 'nothing'} found for {voice})" if voice else "")
                + ("; you see only your own scores" if listener is not None else "") + ".")
    return {"question": question, "route": way if way != "summary" else f"summary:{what}", "voice": voice,
            "absent": absent, "answer": text,
            "citations": [{"id": i, "revision": revision[i], "grade": cited[i]} for i in order[:CITE]],
            "cited": len(order), "cited_ids": order, "top_score": top}


# ------------------------------------------------------------------ the check --
def check(records, questions, answer_key, paraphrases):
    """Each question: declined when the key expects it, answered with citations when not, and every grade-2
    citation in the key's relevant records. Each paraphrase: the route it should take."""
    index = baseline.BM25(records)
    rows = []
    for q in questions:
        a = answer(q["question"], records, index)
        key = answer_key["questions"][q["id"]]
        relevant = set(key["relevant"])
        expect_absent = key["expect"] == "decline"
        cites_ok = all(i in relevant for i in a["cited_ids"]) if a["cited_ids"] else expect_absent
        rows.append({"id": q["id"], "route": a["route"], "expect": key["expect"], "absent": a["absent"],
                     "top_score": a["top_score"], "cited": a["cited"],
                     "citations_in_key": sum(i in relevant for i in a["cited_ids"]),
                     "ok": a["absent"] == expect_absent and cites_ok})
    para = []
    for p in paraphrases:
        a = answer(p["question"], records, index)
        para.append({"question": p["question"], "expect": p["route"], "route": a["route"], "absent": a["absent"],
                     "top_score": a["top_score"], "ok": a["route"] == p["route"] and a["absent"] == p.get("absent", False)})
    answered = [r["top_score"] for r in rows + para if r["route"] == "search" and not r["absent"]]
    declined = [r["top_score"] for r in rows + para if r["route"] == "search" and r["absent"] and r["top_score"] is not None]
    return {"questions": rows, "paraphrases": para, "cutoff": CUTOFF,
            "margin": {"highest_declined": max(declined, default=None), "lowest_answered": min(answered, default=None)}}


def load_paraphrases(path=PARAPHRASES_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))["paraphrases"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Answer a question from the evidence, with citations, or decline.")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--as", dest="account", help="answer as this login account sees the evidence")
    parser.add_argument("--check", action="store_true", help="check the 10 questions and the paraphrases")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    records = evidence.load()
    if args.check:
        from backend.rag import compare
        key = labels.load()
        compare.check_labels(key, safe_files.read_json(evidence.MANIFEST_PATH))
        result = check(records, baseline.load_questions(), key, load_paraphrases())
        for r in result["questions"]:
            said = "declined" if r["absent"] else f"answered, {r['cited']} cited ({r['citations_in_key']} in key)"
            top = f", top {r['top_score']:.2f}" if r["top_score"] is not None else ""
            print(f"  {'ok ' if r['ok'] else 'BAD'} {r['id']} {r['route']:<22} expect {r['expect']:<7} {said}{top}")
        print("Paraphrases:")
        for p in result["paraphrases"]:
            print(f"  {'ok ' if p['ok'] else 'BAD'} {p['route']:<22} (expect {p['expect']}{', declined' if p['absent'] else ''})"
                  f"  {p['question']}")
        m = result["margin"]
        print(f"Cut-off {CUTOFF:g}: highest declined search score {m['highest_declined']}, "
              f"lowest answered {m['lowest_answered']}")
        bad = sum(not r["ok"] for r in result["questions"] + result["paraphrases"])
        print(f"{len(result['questions']) + len(result['paraphrases']) - bad} of "
              f"{len(result['questions']) + len(result['paraphrases'])} as expected.")
        out = evidence.RAG_DIR / "answer_check.json"
        out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Saved: {out}")
        return result
    if not args.question:
        parser.error("ask a question, or use --check")
    listener = None
    if args.account:
        from backend.api import accounts
        acct = accounts.get(args.account)
        if not acct:
            parser.error(f"No account {args.account!r}.")
        if not acct.can("rate_for_others"):                   # the API's rule: only admins see others' scores
            listener = acct.display_name
    a = answer(args.question, records, listener=listener)
    if args.json:
        print(json.dumps(a, indent=1, ensure_ascii=False))
        return a
    print(f"{a['question']}\n[{a['route']}]\n{a['answer']}")
    if a["citations"]:
        print(f"Evidence ({a['cited']} record{'s' if a['cited'] != 1 else ''}"
              + (f", the first {CITE} listed" if a["cited"] > CITE else "") + "):")
        for c in a["citations"]:
            print(f"  {c['id']}  (revision {c['revision']})")
    return a


if __name__ == "__main__":
    main()
