"""read_calls.py | The read-only calls an MCP server makes for one logged-in person.

Builded: plain Python first, with its own checks and tests, before any MCP
or model. Each function makes one GET to the API with the person's own token and returns what the API
answers. The API keeps the last word on who may see what, so nothing here decides permissions.
Later these move into api_calls.py, replacing its stubs.
"""
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

PROJECT = Path(__file__).resolve().parents[2]
RUN_ID = re.compile(r"^[A-Za-z0-9_.-]{1,200}$")
READ_ONLY = ("datasets", "voices", "list_searches", "search_details", "named_genomes", "renders",
             "query_scores", "evidence_question", "search_evidence", "answer_evidence", "results", "about")


class CallError(Exception):
    """A call that was not made, or that the API refused. The message is safe to show."""


@dataclass
class Session:
    api: object                                   # dramatizer_api.Api, or a stand-in in tests
    who: str = ""                                 # the name shown with ratings; never the token
    permissions: frozenset = field(default_factory=frozenset)

    def __repr__(self):
        return f"Session(who={self.who!r})"


def connect(url=None, fingerprint=None):
    """The fingerprint-checked connection from API calls/dramatizer_api.py. Nothing is sent yet."""
    sys.path.insert(0, str(PROJECT / "API calls"))
    from dramatizer_api import Api
    return Api(url or os.environ.get("DRAMATIZER_URL", "https://127.0.0.1:5005"),
               fingerprint or os.environ.get("DRAMATIZER_FINGERPRINT"))


def log_in(api, key=None, user=None):
    """A Session for an API key (DRAMATIZER_KEY), or for an account whose password is in DRAMATIZER_PASSWORD."""
    if key is None and user is None:
        raise CallError("Give an API key (DRAMATIZER_KEY) or an account (DRAMATIZER_USER).")
    if key is None and not os.environ.get("DRAMATIZER_PASSWORD"):
        raise CallError("An account needs DRAMATIZER_PASSWORD: an MCP server cannot ask for it in a prompt.")
    try:
        api.log_in(user=None if key else user, key=key)
    except SystemExit as error:
        raise CallError(str(error)) from None
    me = _get(Session(api=api), "me")
    account = me.get("account") or {}
    return Session(api=api, who=me.get("name") or "", permissions=frozenset(account.get("permissions") or []))


# ----------------------------------------------------------------- the calls
def datasets(session):
    """The datasets, their validation voices, and how many voices the project uses in all."""
    return _get(session, "datasets")


def voices(session, dataset=None, query=None, page=1):
    """One dataset's validation voices, 50 a page. Without a dataset: the datasets. Test voices are never listed."""
    if dataset is None:
        return datasets(session)
    return _get(session, "voices", dataset=dataset, q=query, page=_whole(page, 1, 10**6, "page"))


def list_searches(session, batch=None, pile=None):
    """The genome searches the person may see, newest first; optionally one listening pile."""
    return _get(session, "genome/searches", batch=batch, pile=pile)


def search_details(session, run_id):
    """One search: its genomes, verdicts, ratings and ranking."""
    return _get(session, f"genome/searches/{_run(run_id)}")


def named_genomes(session):
    """The genomes people have picked and named."""
    return _get(session, "genome/named")


def renders(session, query=None, mine=False, page=1):
    """Lines people have spoken or rendered, newest first. mine=True: only this person's."""
    return _get(session, "renders", q=query, mine="1" if mine else None, page=_whole(page, 1, 10**6, "page"))


def query_scores(session, **filters):
    """Listeners' scores, filtered, sorted or grouped. A listener gets only their own; the API decides."""
    return _get(session, "evidence/scores", **filters)


def evidence_question(session, number):
    """One of the six research questions, with the numbers of its chart. Needs the right to see everyone's scores."""
    if not isinstance(number, int) or isinstance(number, bool) or not 1 <= number <= 6:
        raise CallError("The research question is a number from 1 to 6.")
    return _get(session, f"evidence/questions/{number}")


def search_evidence(session, words, limit=5):
    """The evidence records that best match some words, at most 20."""
    return _get(session, "evidence/search", q=_text(words), k=_whole(limit, 1, 20, "limit"))


def answer_evidence(session, question):
    """An answer from the evidence with its record IDs, or 'nothing answers this'."""
    return _get(session, "evidence/answer", q=_text(question))


def results(session, name=None):
    """Baseline and Milestone 2 results. With a name, only the result sets with that name."""
    data = _get(session, "results")
    if name is None:
        return data
    return {"baselines": [b for b in data["baselines"] if b["name"] == name],
            "milestone2": [m for m in data["milestone2"] if m["name"] == name]}


def about(session):
    """What the Dramatizer is, how to use it, and who made it."""
    return _get(session, "about")


# ----------------------------------------------------------------- helpers
def _get(session, route, **params):
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    query = "?" + urlencode(params) if params else ""
    try:
        return session.api.json("GET", "/" + route + query)
    except SystemExit as error:          # dramatizer_api stops a script; a tool gets the reason instead
        raise CallError(str(error)) from None


def _whole(value, low, high, name):
    if not isinstance(value, int) or isinstance(value, bool):
        raise CallError(f"{name} is a whole number.")
    return min(high, max(low, value))


def _text(value, limit=300):
    text = " ".join(str(value or "").split())
    if not text:
        raise CallError("The question is empty.")
    return text[:limit]


def _run(run_id):
    if not isinstance(run_id, str) or not RUN_ID.match(run_id):
        raise CallError("No such search.")
    return run_id

def overview(session):
    """How many genome searches the person may see, how many are rated, and when the newest was made."""
    runs = list_searches(session)["searches"]
    return {"searches": len(runs), "rated": sum(1 for r in runs if r.get("rated")),
            "newest": runs[0]["created"] if runs else None}