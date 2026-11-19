"""dramatizer_mcp.py | MCP server: read-only tools for one logged-in person, over the Dramatizer API.

Start (stdio, one client). Set these first:
    DRAMATIZER_URL, DRAMATIZER_FINGERPRINT, and either DRAMATIZER_KEY or DRAMATIZER_USER with DRAMATIZER_PASSWORD
    python backend/mcp_server/dramatizer_mcp.py

Every tool is a read-only call through read_calls.py, made with this person's own token.
The API decides what they may see. No tool writes, starts a search or speaks a line.
"""
import os
import sys
import json
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import read_calls as rc  # noqa: E402

mcp = FastMCP("dramatizer")
_session = None


def session():
    if _session is None:
        raise RuntimeError("Not logged in: the server could not log in when it started.")
    return _session


def _start():
    global _session
    api = rc.connect()
    _session = rc.log_in(api, key=os.environ.get("DRAMATIZER_KEY"), user=os.environ.get("DRAMATIZER_USER"))


@mcp.tool()
def datasets() -> dict:
    """Use for how many voices and datasets the project has, and how they are split. No arguments."""
    return rc.datasets(session())


@mcp.tool()
def voices(dataset: Optional[str] = None, query: Optional[str] = None, page: int = 1) -> dict:
    """Use to list the validation voices of one dataset (VCTK, CREMA-D or LibriTTS-R), 50 a page. query matches a speaker or words in the line. Test voices are locked and never listed."""
    return rc.voices(session(), dataset=dataset, query=query, page=page)


@mcp.tool()
def list_searches(batch: Optional[str] = None, pile: Optional[int] = None) -> dict:
    """Use to find genome searches and their run_id, newest first. Filter by batch and pile to see one listening pile. Call this before search_details."""
    return rc.list_searches(session(), batch=batch, pile=pile)


@mcp.tool()
def search_details(run_id: str) -> dict:
    """Use for one genome search, by run_id from list_searches: its genomes, verdicts, ratings and ranking."""
    return rc.search_details(session(), run_id)


@mcp.tool()
def named_genomes() -> dict:
    """Use when asked which genomes people have picked and named."""
    return rc.named_genomes(session())


@mcp.tool()
def renders(query: Optional[str] = None, mine: bool = False, page: int = 1) -> dict:
    """Use for lines people have spoken or rendered, newest first. query searches voice, text, genome or maker. mine=True shows only this person's."""
    return rc.renders(session(), query=query, mine=mine, page=page)


@mcp.tool()
def query_scores(listener: Optional[str] = None, voice: Optional[str] = None, dataset: Optional[str] = None,
                 gene_type: Optional[str] = None, genes: Optional[int] = None, genome: Optional[int] = None,
                 clip: Optional[str] = None, pile: Optional[int] = None, verdict: Optional[str] = None,
                 group_by: Optional[str] = None, sort: Optional[str] = None, order: Optional[str] = None,
                 limit: Optional[int] = None) -> dict:
    """Use for who rated what: genome scores filtered by listener, voice, dataset, gene_type, genes (how many), genome (its number), clip (genome, plain or all; genome is the default), pile or verdict. Use group_by (listener, voice, dataset, gene_type, genes, verdict, genome or pile) with sort mean, count or beat_plain for averages and counts. For the six research questions use evidence_question instead."""
    return rc.query_scores(session(), listener=listener, voice=voice, dataset=dataset, gene_type=gene_type,
                           genes=genes, genome=genome, clip=clip, pile=pile, verdict=verdict,
                           group_by=group_by, sort=sort, order=order, limit=limit)


@mcp.tool()
def evidence_question(number: int) -> dict:
     """Use only for the research question about how often a genome beats the plain line by number of active features (question 5), and for the other five research questions by their number. For beating the plain line by dataset, voice or gene type use query_scores with sort beat_plain. Needs a researcher, admin or API key."""
     return rc.evidence_question(session(), number)

@mcp.tool()
def search_evidence(words: str, limit: int = 5) -> dict:
    """Use to see which evidence records match some words (at most 20). Returns record IDs and text, not an answer."""
    return rc.search_evidence(session(), words, limit=limit)


@mcp.tool()
def answer_evidence(question: str) -> dict:
    """Use for questions about the evidence that the other tools do not fit. Returns an answer with its record IDs, or says that nothing in the evidence answers it. If it says so, do not guess."""
    return rc.answer_evidence(session(), question)


@mcp.tool()
def results(name: Optional[str] = None) -> dict:
    """Use for baseline results (WER, similarity, PESQ, STOI, SI-SDR) and Milestone 2 results. Call without name to see every result set, then answer from the list."""
    return rc.results(session(), name=name)


@mcp.tool()
def about() -> dict:
    """Use for what the Dramatizer is, how to use the website, and who made it."""
    return rc.about(session())

@mcp.resource("dramatizer://about")
def about_text() -> str:
    """What the Dramatizer is, how to use it and who made it, as text an assistant can read as context."""
    return json.dumps(rc.about(session()), ensure_ascii=False, indent=2)

@mcp.tool()
def overview() -> dict:
    """Use for how many genome searches exist, how many are rated, and when the newest was made."""
    return rc.overview(session())


if __name__ == "__main__":
    try:
        _start()
    except (rc.CallError, SystemExit) as error:
        print(f"Could not log in: {error}", file=sys.stderr)
        sys.exit(1)
    mcp.run()