# mcp_server/test_graph.py
"""
    python mcp_server/test_graph.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_graph as bg
import dramatizer_mcp as server


def make_project(root):
    """A small fake project: two clips for voiceA, one for voiceB."""
    records = [
        ("voiceA", "clip1", "First sentence."),
        ("voiceA", "clip2", "Second sentence."),
        ("voiceB", "clip3", "Third sentence."),
    ]
    for voice, clip, text in records:
        folder = root / "output" / voice
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{clip}.wav").write_bytes(b"RIFF")  # only checked for existence
        (folder / f"{clip}.json").write_text(json.dumps({
            "target_text": text,
            "reference_audio": f"imported_voices/VCTK/{voice}.wav",
            "reference_text": "reference",
            "voice_name": voice,
            "seconds": 1.0,
            "created": "2026-11-02 00:00:00",
        }), encoding="utf-8")


def test_build_and_serve():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_project(root)

        # build the graph from the fake project
        bg.PROJECT = root
        bg.OUTPUT_DIR = root / "output"
        bg.GRAPH_PATH = root / "history" / "graph.json"
        bg.save(bg.build())
        assert bg.GRAPH_PATH.exists(), "graph was not saved"

        # the server reads that same graph
        server.GRAPH_PATH = bg.GRAPH_PATH
        assert server.list_voices() == ["voiceA", "voiceB"]

        mine = server.generations("voiceA")
        assert sorted(item["text"] for item in mine) == ["First sentence.", "Second sentence."]
        assert all(item["file"].startswith("output/voiceA/") for item in mine)

        assert server.generations("nobody") == []


def test_missing_graph_is_empty():
    server.GRAPH_PATH = Path(tempfile.gettempdir()) / "dramatizer_no_such_graph.json"
    assert server.list_voices() == []
    assert server.generations("voiceA") == []


if __name__ == "__main__":
    tests = [test_build_and_serve, test_missing_graph_is_empty]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")