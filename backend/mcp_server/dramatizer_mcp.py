# backend/mcp_server/dramatizer_mcp.py
"""dramatizer_mcp.py | An MCP server that lets an assistant ask the knowledge graph questions.

Read-only: the tools never generate audio. The only write is the graph file
itself, rebuilt when a new generation has been added since it was saved.

Start it with:  python backend/mcp_server/dramatizer_mcp.py
"""
import json
from pathlib import Path

import networkx as nx
from mcp.server.fastmcp import FastMCP

import build_graph

PROJECT = Path(__file__).resolve().parents[2]
GRAPH_PATH = PROJECT / "history" / "graph.json"
OUTPUT_DIR = PROJECT / "output"

mcp = FastMCP("dramatizer")


def is_stale():
    """True if the graph is missing or older than the newest generated record."""
    if not GRAPH_PATH.exists():
        return True
    newest = max((p.stat().st_mtime for p in OUTPUT_DIR.glob("*/*.json")), default=0)
    return newest > GRAPH_PATH.stat().st_mtime


def load_graph():
    """The graph, rebuilt first if it is out of date."""
    if is_stale():
        build_graph.save(build_graph.build(OUTPUT_DIR, PROJECT), GRAPH_PATH)
    data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    return nx.node_link_graph(data, edges="links")


@mcp.tool()
def list_voices() -> list[str]:
    """List the reference voices that have generated audio."""
    g = load_graph()
    return sorted(d["name"] for _, d in g.nodes(data=True) if d.get("kind") == "voice")


@mcp.tool()
def generations(voice: str) -> list[dict]:
    """List the audio generated for one reference voice: the file and the text it says.

    voice: the name from list_voices, for example "p225_003_mic1".
    """
    g = load_graph()
    return [
        {"file": d["file"], "text": d["text"], "created": d.get("created")}
        for _, d in g.nodes(data=True)
        if d.get("kind") == "output" and d.get("voice") == voice
    ]


if __name__ == "__main__":
    mcp.run()
