# backend/mcp_server/build_graph.py
"""build_graph.py | Build the knowledge graph from the generated clips.

Reads output/<voice>/<clip>.json and its .wav file and makes a graph:
    voice   a reference voice                    (kind="voice")
    output  a generated clip, with the text said (kind="output")

The server rebuilds the graph by itself when it is out of date.
You can also run it by hand:  python backend/mcp_server/build_graph.py
"""
import json
from pathlib import Path

import networkx as nx

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT / "output"
GRAPH_PATH = PROJECT / "history" / "graph.json"


def build(output_dir=OUTPUT_DIR, project=PROJECT):
    g = nx.Graph()

    for info_file in sorted(Path(output_dir).glob("*/*.json")):
        wav = info_file.with_suffix(".wav")
        if not wav.exists():
            continue  # a record without its audio is skipped

        record = json.loads(info_file.read_text(encoding="utf-8"))
        voice = record["voice_name"]
        output = f"output:{wav.name}"

        g.add_node(f"voice:{voice}", kind="voice", name=voice)
        g.add_node(
            output,
            kind="output",
            voice=voice,
            file=wav.relative_to(project).as_posix(),
            text=record["target_text"],
            seconds=record.get("seconds"),
            created=record.get("created"),
        )
        g.add_edge(output, f"voice:{voice}", kind="generated_for")

    return g


def save(g, path=GRAPH_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(g, edges="links")
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    g = build()
    save(g)
    voices = sum(1 for _, d in g.nodes(data=True) if d.get("kind") == "voice")
    outputs = sum(1 for _, d in g.nodes(data=True) if d.get("kind") == "output")
    print(f"{voices} voices, {outputs} generated clips -> {GRAPH_PATH}")
