"""api_calls.py | The API calls an MCP server makes for a logged-in person. NOT IMPLEMENTED YET: stubs only.

This file fixes the shape of the work before it is done: what each call is named, what it takes, which API
route it uses, which permission that route asks for, and what it gives back. Every function raises
NotImplementedError for now. test_api_calls.py checks that each route named here exists in
backend/api/server.py, so the plan stays in step with the API.

Why through the API, not straight to the files
    The MCP server (dramatizer_mcp.py) today reads the knowledge graph on disk and asks no one who is
    calling. Once it can rate, render or see scores, it must obey the same rules as the website: a listener
    sees only their own scores, only a researcher may start a search, and so on. Those rules live in
    server.py (needs_key, needs). So the MCP tools call the API with the person's own token, as the
    website's chatbot does (API calls/chatbot.py), and never read accounts.json or sessions.json themselves.

The two halves
    Logins    connect, log_in / log_in_with_key, who_am_i, change_password, log_out, and a SessionStore
              that keeps one session per MCP client. A password is sent once, to POST /login, and is
              never stored or logged; only the session token (drs_...) or API key (drm_...) is kept, in memory.
    Tools     one function per MCP tool, each taking the Session first. MCP_TOOLS lists them with their route
              and permission, so register() can add them to a FastMCP server and refuse early, with the
              API's own words, what the account may not do.

Planned use (once implemented):
    from backend.mcp_server import api_calls
    conn = api_calls.connect()                                   # DRAMATIZER_URL, DRAMATIZER_FINGERPRINT
    session = api_calls.log_in(conn, "julius", password)         # or log_in_with_key(conn, key)
    api_calls.query_scores(session, listener="Julius", sort="score")
    api_calls.log_out(session)

The connection itself (HTTPS, the certificate checked by fingerprint) is the one in API calls/dramatizer_api.py.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

API = "/api/v1"
SESSION_PREFIX = "drs_"           # a login session (backend/api/accounts.py TOKEN_PREFIX)
KEY_PREFIX = "drm_"               # an API key (backend/api/keys.py)
TODO = "Not implemented yet: api_calls.py only fixes the shape of the MCP and login calls."


# --------------------------------------------------------------------------- errors
class ApiCallError(Exception):
    """The API answered with an error. `status` is the HTTP status; the message is the API's own, safe to show."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class NotLoggedIn(ApiCallError):
    """No session, an ended session, or one the API no longer accepts (401)."""


class NotAllowed(ApiCallError):
    """The account's role does not have the permission the route needs (403)."""


# --------------------------------------------------------------------------- what a login gives
@dataclass
class Connection:
    """Where the API is and how it is reached (url, fingerprint). Holds no login."""
    url: str
    fingerprint: str
    api: Any = None                     # a dramatizer_api.Api, once implemented


@dataclass
class Session:
    """One logged-in person, as seen by the MCP server.

    token        the session token (drs_...) or API key (drm_...); never printed, logged or sent to a model
    kind         "account" or "api key"
    username     None for an API key
    display_name the name kept with ratings
    role         listener, researcher, admin or "api key"
    permissions  what the API allows, from GET /me (an API key is allowed everything)
    expires_at   ISO time a session ends (SESSION_HOURS after login); None for an API key
    """
    connection: Connection
    token: str = field(repr=False)
    kind: str = "account"
    username: Optional[str] = None
    display_name: Optional[str] = None
    role: Optional[str] = None
    permissions: frozenset = frozenset()
    expires_at: Optional[str] = None


# --------------------------------------------------------------------------- logins
def connect(url=None, fingerprint=None):
    """A Connection to the API, without logging in.

    url, fingerprint: as given, else DRAMATIZER_URL and DRAMATIZER_FINGERPRINT. Checks GET /health and that
    the certificate matches the fingerprint; nothing else is sent before that check passes.
    """
    raise NotImplementedError(TODO)


def log_in(connection, username, password):
    """POST /login with a username and password; returns a Session.

    The password is used for this one request and then dropped. Refusals keep the API's message
    (401 wrong username or password, a locked or switched-off account; 429 too many wrong logins).
    Fills the Session from the reply's account and expires_in_hours.
    """
    raise NotImplementedError(TODO)


def log_in_with_key(connection, key):
    """A Session for an API key: checks it with GET /me (no POST /login; a key needs none)."""
    raise NotImplementedError(TODO)


def who_am_i(session):
    """GET /me: the account (name, role, permissions, last login) and the person's jobs. Refreshes the Session."""
    raise NotImplementedError(TODO)


def is_expired(session, now=None):
    """True if a login session has passed expires_at (checked locally, before a call). API keys never expire."""
    raise NotImplementedError(TODO)


def require(session, permission):
    """Raise NotAllowed if the Session lacks `permission` (rate, speak, search, name, rate_for_others).

    Only an early, friendly refusal: the API checks again on every call and has the last word.
    """
    raise NotImplementedError(TODO)


def change_password(session, current, new):
    """POST /me/password {current, new}. The API ends every session of the account, so this one too:
    the person logs in again. Refused for an API key (it has no password)."""
    raise NotImplementedError(TODO)


def log_out(session):
    """POST /logout: ends a login session at the API. For an API key, only forgets it here (the key stays valid)."""
    raise NotImplementedError(TODO)


class SessionStore:
    """The sessions of an MCP server, one per MCP client, in memory only (never written to disk).

    A stdio MCP server has one client; an HTTP one may have several, each with its own login.
    """

    def __init__(self):
        self._sessions = {}

    def put(self, client_id, session):
        """Keep `session` for this client, logging out any session it had before."""
        raise NotImplementedError(TODO)

    def get(self, client_id):
        """The client's Session; NotLoggedIn if there is none or it has expired."""
        raise NotImplementedError(TODO)

    def drop(self, client_id):
        """Log the client out and forget its Session."""
        raise NotImplementedError(TODO)

    def close_all(self):
        """Log every client out (when the MCP server stops)."""
        raise NotImplementedError(TODO)


# --------------------------------------------------------------------------- tools: voices and data
def health(connection):
    """GET /health: {ok, version}. Needs no login."""
    raise NotImplementedError(TODO)


def datasets(session):
    """GET /datasets: the datasets, their voices, and voice_split (validation, locked test, total)."""
    raise NotImplementedError(TODO)


def voices(session, dataset=None):
    """GET /voices?dataset=: the validation voices of one dataset, or of all. Test voices are never listed."""
    raise NotImplementedError(TODO)


def voice_audio(session, dataset, clip_id, save_to):
    """GET /voices/audio: a voice's reference clip, saved as a WAV at save_to; returns the path."""
    raise NotImplementedError(TODO)


# --------------------------------------------------------------------------- tools: genome searches
def list_searches(session):
    """GET /genome/searches: the searches the person may see (a listener: those with their piles)."""
    raise NotImplementedError(TODO)


def search_details(session, run_id):
    """GET /genome/searches/<run_id>: one search, its genomes and the person's own ratings (MOS if allowed)."""
    raise NotImplementedError(TODO)


def search_audio(session, run_id, name, save_to):
    """GET /genome/searches/<run_id>/audio/<name>: one clip of a search, saved as a WAV; returns the path."""
    raise NotImplementedError(TODO)


def start_search(session, dataset, clip_id, text, **settings):
    """POST /genome/searches. Needs "search". Returns the new search's id and its job."""
    raise NotImplementedError(TODO)


def continue_search(session, run_id):
    """POST /genome/searches/<run_id>/continue. Needs "search"."""
    raise NotImplementedError(TODO)


def rate_genomes(session, run_id, scores, artifacts=None):
    """POST /genome/searches/<run_id>/ratings. Needs "rate".

    scores: {"genome_01": 4, ...} for the person themself; the call wraps it as {display_name: scores}.
    Rating for someone else needs "rate_for_others" and is not offered through MCP.
    """
    raise NotImplementedError(TODO)


def listening_piles(session):
    """GET /genome/listening: the listening piles, and which are the person's."""
    raise NotImplementedError(TODO)


def named_genomes(session):
    """GET /genome/named: the genomes people have named."""
    raise NotImplementedError(TODO)


def genome_library(session, voice=None, limit=10):
    """GET /genome/library: the best rated genomes, for one voice or all."""
    raise NotImplementedError(TODO)


# --------------------------------------------------------------------------- tools: speech and renders
def speak(session, dataset, clip_id, text, seed=None):
    """POST /speak. Needs "speak". Queues a plain line; returns the job (see job_state)."""
    raise NotImplementedError(TODO)


def render_genome(session, dataset, clip_id, text, genes, seed=None):
    """POST /genome/render. Needs "speak". Queues a line with a genome; returns the job."""
    raise NotImplementedError(TODO)


def job_state(session, job_id):
    """GET /jobs/<job_id>: queued, running, done or failed, and the place in the queue."""
    raise NotImplementedError(TODO)


def job_audio(session, job_id, save_to):
    """GET /jobs/<job_id>/audio: a finished job's WAV, saved at save_to; returns the path."""
    raise NotImplementedError(TODO)


def renders(session):
    """GET /renders: the lines people rendered on the website (a listener: their own)."""
    raise NotImplementedError(TODO)


# --------------------------------------------------------------------------- tools: evidence and results
def query_scores(session, **filters):
    """GET /evidence/scores: who rated what, filtered, sorted and grouped.

    filters: listener, voice, dataset, gene_type, genes, genome, verdict, pile, search, from, to,
    clip (genome|plain|all), group_by, sort, order, limit. A listener gets only their own scores.
    """
    raise NotImplementedError(TODO)


def evidence_question(session, number):
    """GET /evidence/questions/<number>: research question 1 to 6, with the numbers of its chart.
    Needs the right to see everyone's scores (researcher, admin or API key)."""
    raise NotImplementedError(TODO)


def search_evidence(session, words, limit=5):
    """GET /evidence/search: the records that best match some words."""
    raise NotImplementedError(TODO)


def answer_evidence(session, question):
    """GET /evidence/answer: an answer from the records with their IDs, or "nothing answers this"."""
    raise NotImplementedError(TODO)


def results(session, name=None):
    """GET /results: baseline metrics and Milestone 2 pass rates, or one result set by name."""
    raise NotImplementedError(TODO)


def about(session):
    """GET /about: what the Dramatizer is, how to use it, and who made it."""
    raise NotImplementedError(TODO)


def ask_assistant(session, question, history=None):
    """POST /assistant/ask {question, history}: the website's chatbot, answering as this person."""
    raise NotImplementedError(TODO)


# --------------------------------------------------------------------------- the MCP side
@dataclass(frozen=True)
class Tool:
    """One MCP tool: the function, the API route it calls, and the permission it needs (None: any login)."""
    function: Callable
    method: str
    route: str
    permission: Optional[str] = None
    writes: bool = False            # changes something (a rating, a search, a render); MCP clients may ask first


MCP_TOOLS = {
    # logins (the session is the MCP client's; these are offered as tools too)
    "who_am_i": Tool(who_am_i, "GET", "/me"),
    "change_password": Tool(change_password, "POST", "/me/password", writes=True),
    "log_out": Tool(log_out, "POST", "/logout", writes=True),
    # voices and data
    "datasets": Tool(datasets, "GET", "/datasets"),
    "voices": Tool(voices, "GET", "/voices"),
    "voice_audio": Tool(voice_audio, "GET", "/voices/audio"),
    # genome searches
    "list_searches": Tool(list_searches, "GET", "/genome/searches"),
    "search_details": Tool(search_details, "GET", "/genome/searches/<run_id>"),
    "search_audio": Tool(search_audio, "GET", "/genome/searches/<run_id>/audio/<name>"),
    "start_search": Tool(start_search, "POST", "/genome/searches", "search", writes=True),
    "continue_search": Tool(continue_search, "POST", "/genome/searches/<run_id>/continue", "search", writes=True),
    "rate_genomes": Tool(rate_genomes, "POST", "/genome/searches/<run_id>/ratings", "rate", writes=True),
    "listening_piles": Tool(listening_piles, "GET", "/genome/listening"),
    "named_genomes": Tool(named_genomes, "GET", "/genome/named"),
    "genome_library": Tool(genome_library, "GET", "/genome/library"),
    # speech and renders
    "speak": Tool(speak, "POST", "/speak", "speak", writes=True),
    "render_genome": Tool(render_genome, "POST", "/genome/render", "speak", writes=True),
    "job_state": Tool(job_state, "GET", "/jobs/<job_id>"),
    "job_audio": Tool(job_audio, "GET", "/jobs/<job_id>/audio"),
    "renders": Tool(renders, "GET", "/renders"),
    # evidence and results
    "query_scores": Tool(query_scores, "GET", "/evidence/scores"),
    "evidence_question": Tool(evidence_question, "GET", "/evidence/questions/<int:number>"),
    "search_evidence": Tool(search_evidence, "GET", "/evidence/search"),
    "answer_evidence": Tool(answer_evidence, "GET", "/evidence/answer"),
    "results": Tool(results, "GET", "/results"),
    "about": Tool(about, "GET", "/about"),
    "ask_assistant": Tool(ask_assistant, "POST", "/assistant/ask"),
}
LOGIN_ROUTES = {"log_in": ("POST", "/login"), "health": ("GET", "/health")}


def call_tool(store, client_id, name, arguments):
    """Run MCP tool `name` for a client: its Session from the store, require() its permission, call it.

    Returns plain JSON-ready data. NotLoggedIn and NotAllowed become MCP tool errors with the API's message;
    a token never appears in a result or an error.
    """
    raise NotImplementedError(TODO)


def register(mcp, store, client_id="stdio"):
    """Add MCP_TOOLS, plus a log_in tool that puts a Session in the store, to a FastMCP server.

    The log_in tool takes a username and asks for the password through the MCP client's own prompt
    (elicitation), or reads DRAMATIZER_PASSWORD / DRAMATIZER_KEY for a server started by hand; a password
    never passes through the model's conversation.
    """
    raise NotImplementedError(TODO)
