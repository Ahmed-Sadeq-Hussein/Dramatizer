# mcp_server/dramatizer_mcp.py
"""dramatizer_mcp.py | An MCP server that lets an assistant ask the knowledge graph questions.

Start it with:  python mcp_server/dramatizer_mcp.py
"""
import json
from pathlib import Path

import networkx as nx
from mcp.server.fastmcp import FastMCP

GRAPH_PATH = Path(__file__).resolve().parent.parent / "history" / "graph.json"

mcp = FastMCP("dramatizer")


def load_graph():
    """The graph saved by build_graph.py. Empty if it has not been built yet."""
    if not GRAPH_PATH.exists():
        return nx.Graph()
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