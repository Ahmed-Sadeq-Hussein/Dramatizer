"""API_REQUEST.py | Ask the Dramatizer API about its results, as small JSON answers a chatbot can use.

Each function below asks the running API (backend/api/server.py) one question and returns plain Python
data (dicts and lists, ready for json.dumps): the genome searches and how their genomes were rated, the
best genomes, the listening piles, people's renders, and the measured results (baselines, Milestone 2).
Heavy fields (audio checks, every measurement) are left out, so an answer fits in a language model's prompt.

For a chatbot (e.g. Gemma 4 with function calling, through Ollama, vLLM or llama.cpp):
    TOOLS              the functions described as JSON schemas, to give the model as its tools
    run_tool(name, args) runs the one the model picked and returns its answer as a JSON string

    from API_REQUEST import Results, TOOLS
    results = Results.connect()                 # address, fingerprint and login from the environment
    answer = results.run_tool("top_genomes", {"voice": "CREMA-D/1014", "limit": 3})

From the command line:
    python API_REQUEST.py overview
    python API_REQUEST.py top_genomes '{"limit": 5}'
    python API_REQUEST.py tools                 # the tool schemas, as JSON

Who you are and where the API is, as for dramatizer_api.py (README.md here):
    DRAMATIZER_URL          e.g. https://127.0.0.1:5005, or the playit address
    DRAMATIZER_FINGERPRINT  the API certificate's fingerprint (python main_backend.py api --fingerprint)
    DRAMATIZER_USER         an account (the password is asked for, or read from DRAMATIZER_PASSWORD)
    DRAMATIZER_KEY          or an API key instead of an account

What an answer holds depends on the account: a listener sees only their own scores and no MOS (they rate
blind); a researcher, an admin or an API key sees everyone's MOS. A chatbot that reports rankings should
use a researcher account or an API key. Only Python's standard library is needed.
"""
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote, urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dramatizer_api import Api  # noqa: E402  (the fingerprint-checked connection)


class ApiError(Exception):
    """The API refused or could not answer: the message says why."""


def _num(x, digits=3):
    return None if x is None else round(float(x), digits)


def _mos(x):
    """A MOS as a number: the API gives {mos, n, low, high, scores} where the caller may see it."""
    return _num(x.get("mos")) if isinstance(x, dict) else _num(x)


def _listeners(x):
    return x.get("n") if isinstance(x, dict) else None


def _genome(g):
    """One genome, without the audio checks and measurements: what it does and how it did."""
    sanity = g.get("sanity") or {}
    return {"number": g.get("number"), "genes": g.get("described") or [x.get("feature") for x in g.get("genes", [])],
            "spoken_as": g.get("markup"), "verdict": g.get("verdict"),
            "failed_checks": [k[:-3] for k in ("wer_ok", "similarity_ok", "pace_ok") if sanity.get(k) is False],
            "mos": _mos(g.get("mos")), "listeners": _listeners(g.get("mos")),
            "above_plain_line": _num(g.get("above_baseline")), "rank": g.get("rank"),
            "picked_as": g.get("picked"), "audio": g.get("audio")}


class Results:
    """The questions about results, asked of one API with one login."""

    def __init__(self, api):
        self.api = api

    # ------------------------------------------------------------- connecting --
    @classmethod
    def connect(cls, url=None, fingerprint=None, user=None, key=None):
        """Log in (arguments, or the DRAMATIZER_* environment variables) and return a Results."""
        api = Api(url or os.environ.get("DRAMATIZER_URL", "https://127.0.0.1:5005"),
                  fingerprint or os.environ.get("DRAMATIZER_FINGERPRINT"))
        key = key or os.environ.get("DRAMATIZER_KEY")
        user = user or os.environ.get("DRAMATIZER_USER")
        if not (key or user):
            raise ApiError("Give an account (DRAMATIZER_USER) or an API key (DRAMATIZER_KEY).")
        api.log_in(user=None if key else user, key=key)
        results = cls(api)
        results._user = None if key else user
        return results

    def close(self):
        """Log out (an API key stays valid)."""
        self.api.log_out(getattr(self, "_user", None))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _get(self, route, **params):
        params = {k: v for k, v in params.items() if v not in (None, "", False)}
        path = route + ("?" + urlencode(params) if params else "")
        try:
            return self.api.json("GET", path)
        except SystemExit as error:            # dramatizer_api stops a script; a chatbot gets the reason instead
            raise ApiError(str(error)) from None

    def _post(self, route, body):
        try:
            return self.api.json("POST", route, body)
        except SystemExit as error:
            raise ApiError(str(error)) from None

    # -------------------------------------------------------------- questions --
    def overview(self):
        """How much there is: searches (rated or not), listening piles, named genomes, renders, result sets."""
        searches = self._get("genome/searches")
        totals = searches.get("totals") or {}            # every search; the list itself stops at 100
        results = self._get("results")
        me = self._get("me")
        data = self._get("datasets")
        return {"you": me.get("name"), "role": (me.get("account") or {}).get("role"),
                "voices_in_the_project": data.get("voice_split"),
                "validation_voices_per_dataset": {d["name"]: d["speakers"] for d in data["datasets"]},
                "sees_everyones_mos": bool({"search", "rate_for_others"} & set((me.get("account") or {}).get("permissions") or [])),
                "searches": totals.get("searches", len(searches["searches"])),
                "rated_searches_among_the_newest_100": sum(s["rated"] for s in searches["searches"]),
                "voices_with_genome_searches": totals.get("voices", len({s["voice"] for s in searches["searches"]})),
                "genomes_made_in_searches": totals.get("genomes"),
                "genomes_that_passed_the_checks": totals.get("passed_checks"),
                "listening_piles": searches["piles"], "named_genomes": len(self._get("genome/named")["named"]),
                "renders": self._get("renders")["total"],
                "baselines": [b["name"] for b in results["baselines"]],
                "milestone2_runs": [m["name"] for m in results["milestone2"]],
                "jobs_waiting_on_the_host": me.get("jobs_waiting")}

    def list_searches(self, voice=None, batch=None, pile=None, rated=None, limit=50):
        """Genome searches, newest first: voice, line, status, how many genomes passed, whether rated."""
        data = self._get("genome/searches", batch=batch, pile=pile)
        out = []
        for s in data["searches"]:
            if voice and voice.lower() not in s["voice"].lower():
                continue
            if rated is not None and bool(s["rated"]) != bool(rated):
                continue
            summary = s.get("summary") or {}
            out.append({"id": s["id"], "voice": s["voice"], "line": s["text"], "made": s["created"],
                        "status": s["status"], "genomes": summary.get("total"), "passed_checks": summary.get("fine"),
                        "failed_checks": summary.get("terrible"), "rated": s["rated"],
                        "pile": f"{s['batch']} pile {s['pile']}" if s.get("batch") else None})
        return {"searches": out[:limit], "total": len(out)}

    def search_details(self, run_id):
        """One search: its line and voice, and every genome ranked by listeners' scores (MOS) where rated."""
        data = self._get(f"genome/searches/{quote(run_id)}")
        run, ranking = data["run"], data["ranking"] or {}
        ranked = ranking.get("genomes") or run.get("genomes") or []
        plain = ranking.get("baseline") or {}
        return {"id": run["id"], "voice": run["voice"], "line": run["text"], "method": run.get("method_label") or run.get("method"),
                "made": run.get("created"), "state": data["state"], "listeners": ranking.get("raters") or [],
                "plain_line_mos": _mos(plain),
                "genomes": [_genome(g) for g in ranked]}

    def top_genomes(self, voice=None, limit=5, rated_only=True):
        """The best genomes across searches by MOS (highest first), optionally for one voice."""
        searches = self.list_searches(voice=voice, rated=True if rated_only else None, limit=1000)["searches"]
        found = []
        for s in searches:
            details = self.search_details(s["id"])
            for g in details["genomes"]:
                if g["mos"] is not None:
                    found.append({"search": s["id"], "voice": s["voice"], "line": s["line"], **g})
        found.sort(key=lambda g: g["mos"], reverse=True)
        note = None if found else ("No MOS to rank by: nothing is rated yet, or this account rates blind "
                                   "(a listener). Use a researcher account or an API key.")
        return {"genomes": found[:limit], "considered": len(found), "note": note}

    def named_genomes(self):
        """The genomes listeners picked and named, with their genes and MOS."""
        named = self._get("genome/named")["named"]
        return {"named": [{"name": e.get("name"), "voice": e.get("voice"), "line": e.get("text"),
                           "genes": e.get("described") or [x.get("feature") for x in e.get("genes", [])],
                           "spoken_as": e.get("markup"), "mos": _mos(e.get("mos")),
                           "from_search": e.get("run"), "genome": e.get("genome"), "named_by": e.get("named_by"),
                           "made_by_hand": bool(e.get("edited"))} for e in named]}

    def listening_piles(self):
        """Which listening piles exist, and which this account is assigned to rate."""
        data = self._get("genome/listening")
        return {"piles": data.get("piles"), "assigned_to_you": data.get("assigned"), "you": data.get("rater")}

    def pile(self, batch, pile, voice=None):
        """One listening pile: its voices, and every clip with this account's score (and the MOS if allowed)."""
        data = self._get("genome/listening", batch=batch, pile=pile)
        keep = lambda v: not voice or voice.lower() in str(v).lower()
        return {"batch": data["batch"], "pile": data["pile"], "you": data.get("rater"),
                "sees_everyones_mos": data.get("sees_others"),
                "voices": [{"search": r["id"], "voice": r["voice"], "line": r["text"], "clips": r.get("clips"),
                            "you_scored": r.get("mine"), "listeners": r.get("raters")}
                           for r in data["runs"] if keep(r["voice"])],
                "clips": [{"search": c["run"], "voice": c["voice"], "clip": c["clip"],
                           "genes": "; ".join(c.get("described") or []) or "none (the plain line)",
                           "verdict": c.get("verdict"), "your_score": c.get("mine"), "mos": _mos(c.get("mos")),
                           "mos_above_plain_line": _num(c.get("mos_above_baseline")),
                           "yours_above_plain_line": _num(c.get("mine_above_baseline"))}
                          for c in data["clips"] if keep(c["voice"])]}

    def renders(self, query=None, mine=False, page=1):
        """Lines people spoke or rendered (newest first, 30 a page): voice, line, genome, who, when, your own score."""
        data = self._get("renders", q=query, mine="1" if mine else None, page=page)
        return {"renders": [{k: r.get(k) for k in ("id", "voice", "text", "genome", "genes", "markup", "by", "made",
                                                   "seed", "my_score")} for r in data["renders"]],
                "total": data["total"], "page": data["page"], "pages": data["pages"]}

    def baseline_results(self, name=None):
        """The measured baselines (plain CosyVoice 3, no genome): mean WER, similarity, PESQ, STOI, SI-SDR, checks."""
        out = []
        every_baseline = self._get("results")["baselines"]
        names = [b["name"] for b in every_baseline]
        note = None
        if name and name not in names:            # e.g. a metric's name: every baseline, which has every metric
            note = f"No baseline is called {name!r} (they are {', '.join(names)}): all of them are shown."
            name = None
        for b in every_baseline:
            if name and b["name"] != name:
                continue
            every = (b.get("summary") or {}).get("all") or {}
            out.append({"name": b["name"], "made": b.get("made"), "voices": b.get("voices"), "lines": b.get("lines"),
                        "git_commit": b.get("git_commit"),
                        "metrics": {m: {"mean": _num(v.get("mean"), 4), "low": _num(v.get("low"), 4),
                                        "high": _num(v.get("high"), 4), "clips": v.get("n")}
                                    for m, v in every.items() if isinstance(v, dict) and "mean" in v},
                        "checks_passed": sum(c["passed"] for c in b.get("checks", [])), "checks": len(b.get("checks", []))})
        return {"baselines": out, **({"note": note} if note else {})}

    def milestone2_results(self, name=None):
        """Milestone 2 (random genome search against the baseline): share of genomes that passed, and why others failed."""
        out = []
        for m in self._get("results")["milestone2"]:
            if name and m["name"] != name:
                continue
            shares = (m.get("summary") or {}).get("shares") or {}
            out.append({"name": m["name"],
                        "by_group": {group: {"genomes": s.get("genomes"), "passed": _num(s.get("fine")),
                                             "failed_because": {k: _num(v) for k, v in (s.get("terrible_because") or {}).items()}}
                                     for group, s in shares.items()},
                        "other": {k: v for k, v in (m.get("summary") or {}).items() if k != "shares"}})
        return {"milestone2": out}

    def voices(self, dataset=None, query=None, page=1):
        """The validation voices in the data (50 a page): speaker, dataset and what the clip says."""
        if not dataset:
            data = self._get("datasets")
            return {"datasets": data["datasets"], "voice_split": data.get("voice_split")}
        data = self._get("voices", dataset=dataset, q=query, page=page)
        return {"voices": [{"speaker": c["speaker"], "dataset": c["dataset"], "clip_id": c["clip_id"], "text": c["text"]}
                           for c in data["clips"]], "total": data.get("total"), "page": data.get("page"),
                "pages": data.get("pages")}

    # ------------------------------------------------- the evidence (scores, six questions) --
    def query_scores(self, listener=None, voice=None, dataset=None, gene_type=None, genes=None, genome=None, verdict=None,
                     pile=None, clip="genome", group_by=None, sort=None, order="desc", limit=10,
                     from_date=None, to_date=None, min_score=None, max_score=None):
        """Listeners' scores, filtered, then sorted or grouped (means, counts, share better than the plain line)."""
        data = self._get("evidence/scores", listener=listener, voice=voice, dataset=dataset, gene_type=gene_type,
                         genes=genes, genome=genome, verdict=verdict, pile=pile, clip=clip, group_by=group_by, sort=sort,
                         order=order, limit=limit, min_score=min_score, max_score=max_score,
                         **{"from": from_date, "to": to_date})
        for row in data.get("rows", []):
            row.pop("spoken_as", None)            # "search" stays: the page links each genome to the studio by it
        return data

    def evidence_question(self, number, from_date=None, to_date=None):
        """One of the six genome questions, answered as its chart answers it (the same numbers)."""
        return self._get(f"evidence/questions/{int(number)}", **{"from": from_date, "to": to_date})

    def evidence_status(self, voice=None):
        """The analysis's conclusions and how sure they are: snapshot_id, ratings up to, rule, coverage, per voice
        a winner only with enough evidence (else why not), artifact checks, caveats."""
        return self._get("evidence/snapshot", voice=voice)

    def define(self, term=None):
        """What a word or measurement means in the Dramatizer, how to read its number and where it is used."""
        return self._get("glossary", q=term)

    def render_info(self, render_id):
        """One render by its ID (as Copy ID copies it, "render:..."): voice, line, genome, genes, who made it."""
        r = self._get(f"renders/{quote(str(render_id).strip(), safe='')}")["render"]
        return {"id": r["id"], "short_id": r.get("short_id"), "name": r.get("name"), "voice": r.get("voice"),
                "line": r.get("text"),
                "genome": r.get("genome") or ("edited genome" if r.get("genes") else "plain line (no genome)"),
                "genes": r.get("genes") or [], "spoken_as": r.get("markup"), "speed": r.get("speed"),
                "seed": r.get("seed"), "made_by": r.get("by"), "made": r.get("made"),
                "render_seconds": r.get("seconds")}

    MEASURE_SECONDS = 50          # how long measure_render waits for the GPU job before handing back its job id

    def measure_render(self, render_id, compare_with=None):
        """A render measured on the host (the Lab's measurements): against its voice's original recording, and
        against another render when one is given."""
        first = self._get(f"renders/{quote(str(render_id).strip(), safe='')}")["render"]
        recordings, names = [{"kind": "render", "id": first["id"]}], ["the render"]
        if compare_with:
            other = self._get(f"renders/{quote(str(compare_with).strip(), safe='')}")["render"]
            recordings.append({"kind": "render", "id": other["id"]})
            names.append("the other render")
        reference = None
        if first.get("clip_id") and "/" in str(first.get("voice") or ""):
            recordings.append({"kind": "voice", "dataset": first["voice"].split("/", 1)[0], "clip_id": first["clip_id"]})
            names.append("the original recording")
            reference = len(recordings) - 1
        job = self._post("lab/compare", {"recordings": recordings, "reference": reference})["job"]
        waited = 0.0
        while job["status"] in ("queued", "running") and waited < self.MEASURE_SECONDS:
            time.sleep(2)
            waited += 2
            job = self._get(f"jobs/{job['id']}")["job"]
        if job["status"] == "failed":
            raise ApiError(f"The measurement failed: {job.get('error')}")
        if job["status"] != "done":
            return {"note": f"Still measuring on the host (the job is {job['status']}). The result will be in the "
                            "Lab tab under 'Your earlier comparisons' when it is done.", "job": job["id"]}
        lab = job["lab"]
        rows = []
        for name, row in zip(names, lab["recordings"]):
            quality = row.get("quality") or {}
            rows.append({"which": name, "label": row["label"], "wer": row.get("wer"),
                         "wrong_words": f"{row['errors']} of {row['word_count']}" if row.get("word_count") else None,
                         "script": row.get("text"), "heard": row.get("heard"),
                         "similarity_to_original": row.get("similarity"), "pesq": quality.get("pesq"),
                         "stoi": quality.get("stoi"), "si_sdr_db": quality.get("si_sdr"), "seconds": row.get("seconds"),
                         "speech_seconds": row.get("speech_seconds"), "words_per_second": row.get("words_per_second")})
        out = {"render_id": first["id"], "voice": first.get("voice"), "recordings": rows,
               "limits": lab.get("thresholds"), "models": lab.get("models"), "measured_in_seconds": lab.get("seconds")}
        if compare_with and lab.get("matrix"):
            out["similarity_between_the_two_renders"] = lab["matrix"][0][1]
        if reference is None:
            out["note"] = "The render's original recording is not known, so there is no similarity to it."
        return out

    def about_dramatizer(self):
        """What the Dramatizer is, how to use the website, and who made it (the Home page's About)."""
        return self._get("about")

    def answer_evidence(self, question):
        """An answer from the evidence with the record IDs it rests on, or a plain "nothing answers this"."""
        return self._get("evidence/answer", q=question)

    def search_evidence(self, query, kind=None, k=6):
        """The evidence records (scores, search rankings, sanity checks) that best match a question."""
        return self._get("evidence/search", q=query, kind=kind, k=k)

    # ----------------------------------------------------------- for a chatbot --
    def run_tool(self, name, arguments=None):
        """Run one of TOOLS by name with the model's arguments; the answer (or the error) as a JSON string."""
        if name not in TOOL_NAMES:
            return json.dumps({"error": f"No tool {name!r}. The tools are: {', '.join(TOOL_NAMES)}."})
        if isinstance(arguments, str):
            arguments = json.loads(arguments or "{}")
        arguments = dict(arguments or {})
        known = set(next(t for t in TOOLS if t["function"]["name"] == name)["function"]["parameters"]["properties"])
        ignored = sorted(set(arguments) - known)          # a model may invent an argument: drop it, and say so
        arguments = {k: v for k, v in arguments.items() if k in known}
        try:
            answer = getattr(self, name)(**arguments)
        except (ApiError, TypeError, ValueError) as error:
            return json.dumps({"error": str(error), "allowed_arguments": sorted(known)})
        if ignored and isinstance(answer, dict):
            answer = {"ignored_arguments": ignored, "allowed_arguments": sorted(known), **answer}
        return json.dumps(answer, ensure_ascii=False)


def _tool(name, description, properties=None, required=()):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties or {}, "required": list(required)}}}


STRING, INTEGER, BOOLEAN = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}
TOOLS = [
    _tool("overview", "How much data there is, as counts: how many genome searches (and rated ones), the voices in "
          "the project (total, validation, locked test, per dataset), listening piles, named genomes, renders, and "
          "which baseline and Milestone 2 results exist, and how many genomes the searches made (and how many passed "
          "the checks). Use it for 'how many genomes/searches/voices/renders'."),
    _tool("list_searches", "List genome searches (newest first, a page at a time) with voice, line, status and how "
          "many genomes passed the automatic checks. Not for counting them: use overview.", {
              "voice": {**STRING, "description": "Part of a voice name, e.g. 'CREMA-D/1014' or '1014'"},
              "batch": {**STRING, "description": "A listening batch, e.g. 'mos-diverse-v1'"},
              "pile": {**INTEGER, "description": "A pile number in that batch"},
              "rated": {**BOOLEAN, "description": "Only rated (true) or only unrated (false) searches"},
              "limit": {**INTEGER, "description": "At most this many (default 50)"}}),
    _tool("search_details", "One genome search: its line and voice, and every genome with its genes, automatic "
          "verdict and listener score (MOS), best first.", {
              "run_id": {**STRING, "description": "The search id from list_searches"}}, ["run_id"]),
    _tool("top_genomes", "The single best-rated genomes by MOS, optionally for one voice. Not for averages per voice, "
          "dataset or gene type: use query_scores with group_by.", {
              "voice": {**STRING, "description": "Part of a voice name"},
              "limit": {**INTEGER, "description": "How many (default 5)"}}),
    _tool("named_genomes", "The genomes listeners picked and named, with their genes and MOS."),
    _tool("listening_piles", "Which listening piles exist and which are assigned to this account."),
    _tool("pile", "One listening pile: its voices, and every clip with this account's score and the MOS. "
          "A whole pile is long (about 90 clips): give a voice to keep the answer short.", {
              "batch": {**STRING, "description": "e.g. 'mos-diverse-v1'"}, "pile": INTEGER,
              "voice": {**STRING, "description": "Only this voice (part of its name)"}}, ["batch", "pile"]),
    _tool("renders", "Lines people spoke or rendered in the website, newest first, with genome and who made them.", {
              "query": {**STRING, "description": "Words to find in the voice, line, genome or maker"},
              "mine": {**BOOLEAN, "description": "Only this account's renders"},
              "page": INTEGER}),
    _tool("baseline_results", "The measured baselines (plain voice cloning, no genome): mean WER, speaker "
          "similarity, PESQ, STOI and SI-SDR with confidence ranges.", {
              "name": {**STRING, "description": "One baseline version, e.g. 'v4' (not a metric: every baseline has "
                                                "every metric); all when left out"}}),
    _tool("milestone2_results", "Milestone 2: how many random-search genomes passed the checks against the baseline, "
          "by group (short or long lines, dataset), and why the others failed.", {
              "name": {**STRING, "description": "One run, e.g. 'random-v2'; all when left out"}}),
    _tool("query_scores", "Listeners' scores of genomes (0 to 5), filtered and then either listed (sorted) or grouped "
          "with mean, count, range, mean above the plain line and share better than the plain line. Use it for "
          "who rated what, a listener's highest or lowest, gene types, voices, datasets, number of genes.", {
              "listener": {**STRING, "description": "A listener's name, e.g. 'Rama', 'Ash', 'Julius'"},
              "voice": {**STRING, "description": "Part of a voice name, e.g. 'CREMA-D/1014' or 'p250'"},
              "dataset": {**STRING, "description": "VCTK, CREMA-D or LibriTTS-R"},
              "gene_type": {**STRING, "description": "A gene type, e.g. 'sigh', 'speed', 'pause (dramatic)', 'laughing speech'"},
              "genes": {**INTEGER, "description": "Only genomes with this many genes"},
              "genome": {**INTEGER, "description": "One genome's number within its voice's search (use with voice)"},
              "verdict": {**STRING, "description": "Sanity verdict: 'fine' or 'terrible'"},
              "pile": INTEGER,
              "clip": {**STRING, "enum": ["genome", "plain", "all"], "description": "genome (default), plain line, or both"},
              "group_by": {**STRING, "enum": ["listener", "voice", "dataset", "gene_type", "genes", "verdict", "genome", "pile"]},
              "sort": {**STRING, "enum": ["score", "above_plain", "rated_on", "mean", "mean_above_plain", "count", "beat_plain"]},
              "order": {**STRING, "enum": ["desc", "asc"], "description": "desc = highest first (default), asc = lowest first"},
              "limit": {**INTEGER, "description": "How many rows or groups (default 10; 20 for a list)"},
              "min_score": {**INTEGER, "description": "Only scores of at least this (e.g. 5: 'rated 5', 4: 'did well')"},
              "max_score": {**INTEGER, "description": "Only scores of at most this"},
              "from_date": {**STRING, "description": "Scores given from this date, YYYY-MM-DD"},
              "to_date": {**STRING, "description": "Scores given up to this date, YYYY-MM-DD"}}),
    _tool("evidence_question", "The six research questions, answered exactly as their charts answer them: "
          "1 best genomes per voice, 2 which genomes produce artifacts, 3 configurations with bad MOS, "
          "4 configurations that are consistently good, 5 how often a genome beats the plain line by number of "
          "genes, 6 which genomes (gene types) to exclude.", {
              "number": {**INTEGER, "description": "1 to 6"},
              "from_date": {**STRING, "description": "YYYY-MM-DD"}, "to_date": {**STRING, "description": "YYYY-MM-DD"}},
          ["number"]),
    _tool("evidence_status", "What the analysis concluded and how sure it is: the snapshot_id and the newest rating "
          "it used, the aggregation rule, coverage (genomes rated, listeners per genome), per voice whether there is a "
          "winner or not enough evidence and why, the artifact checks, and caveats. Use it for 'which genome should I "
          "use', 'is there a winner', 'how sure are we', 'which ratings is this based on'.", {
              "voice": {**STRING, "description": "One voice, DATASET/SPEAKER, e.g. VCTK/p250; all when left out"}}),
    _tool("define", "What a word or measurement means in the Dramatizer, how to read its number (its scale and the "
          "project's limits) and where the program uses it. Measurements: WER, speaker similarity, PESQ, STOI, "
          "SI-SDR, SQUIM, speech seconds, words per second, pace, MOS, above plain, beat the plain line, confidence "
          "interval, sanity check (verdict fine/terrible), artifact check, different person, reward. Ideas: genome, "
          "gene and every gene type (sigh, pause, laughing speech, accent, emphasis, speed...), plain line, baseline, "
          "reference recording, voice cloning, render, seed, markup, genome search, named genome, listening pile, "
          "blind, voice, validation and test voices, custom voice, evidence, milestone 2, the Lab. Use it for "
          "'what is X', 'what does X mean / measure / stand for', 'explain X'. Without a term: every term.", {
              "term": {**STRING, "description": "The word to explain, e.g. 'STOI', 'plain line', 'sanity check'"}}),
    _tool("render_info", "One render (a line spoken in the website, listed in Listen > Renders) by its ID, as its "
          "Copy ID button copies it, e.g. 'render:20261119_220500_543947': its voice, line, genome and every gene, "
          "how it was spoken (markup), seed, who made it and when.", {
              "render_id": {**STRING, "description": "The render's ID exactly as given, with or without 'render:'"}},
          ["render_id"]),
    _tool("measure_render", "Measure a render on the host: speaker similarity to its voice's original recording "
          "(the reference), word error rate with what was heard, PESQ, STOI, SI-SDR, length and speaking rate, for "
          "the render and the original. With compare_with (another render's ID): also how alike the two renders "
          "sound. Use it for 'how similar is this render to its reference', 'what is the WER of render X', "
          "'compare these two renders'. It runs on the GPU and takes 10 to 60 seconds.", {
              "render_id": {**STRING, "description": "The render's ID exactly as given, with or without 'render:'"},
              "compare_with": {**STRING, "description": "Another render's ID, to compare the two"}},
          ["render_id"]),
    _tool("about_dramatizer", "What the Dramatizer is, how to use the website (Rate, Generate, Listen, Ask, "
          "Research), and the project's contributors."),
    _tool("answer_evidence", "Answer a question from the evidence records, citing their IDs, or say that nothing "
          "in the evidence answers it. Counts over many records (best, bad, better than the plain line, artifacts, "
          "exclude), one genome of one voice, or a search with a cut-off.", {"question": STRING}, ["question"]),
    _tool("search_evidence", "Find the evidence records that best match a question in words: single scores, "
          "searches' rankings, sanity checks. For questions the other tools do not fit.", {
              "query": STRING, "kind": {**STRING, "enum": ["clip", "run", "sanity"]},
              "k": {**INTEGER, "description": "How many records (default 6)"}}, ["query"]),
    _tool("voices", "Without a dataset: how many voices the project uses (validation and locked test voices) and "
          "each dataset's validation voices. With one: its voices and clips.", {
              "dataset": STRING, "query": STRING, "page": INTEGER}),
]
TOOL_NAMES = [t["function"]["name"] for t in TOOLS]


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("Tools:", ", ".join(TOOL_NAMES))
        return 0
    if argv[0] == "tools":
        print(json.dumps(TOOLS, indent=2))
        return 0
    with Results.connect() as results:
        answer = results.run_tool(argv[0], argv[1] if len(argv) > 1 else None)
    print(json.dumps(json.loads(answer), indent=2, ensure_ascii=False))
    return 1 if '"error"' in answer[:12] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
