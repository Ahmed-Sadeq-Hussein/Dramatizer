# backend/mcp_server/test_graph.py
"""test_graph.py | Check the graph and the tools without a language model.

    python backend/mcp_server/test_graph.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dramatizer_mcp as server


def add_clip(root, voice, clip, text):
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


def point_server_at(root):
    """Use a temporary project, so the real history/ and output/ are never touched."""
    server.PROJECT = root
    server.OUTPUT_DIR = root / "output"
    server.GRAPH_PATH = root / "history" / "graph.json"


def test_builds_graph_when_missing():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        add_clip(root, "voiceA", "clip1", "First sentence.")
        add_clip(root, "voiceA", "clip2", "Second sentence.")
        add_clip(root, "voiceB", "clip3", "Third sentence.")
        point_server_at(root)

        assert server.list_voices() == ["voiceA", "voiceB"]
        assert server.GRAPH_PATH.exists(), "graph was not saved"
        mine = server.generations("voiceA")
        assert sorted(item["text"] for item in mine) == ["First sentence.", "Second sentence."]
        assert all(item["file"].startswith("output/voiceA/") for item in mine)
        assert server.generations("nobody") == []


def test_rebuilds_when_a_new_clip_is_added():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        add_clip(root, "voiceB", "clip3", "Third sentence.")
        point_server_at(root)
        assert len(server.generations("voiceB")) == 1

        # make the saved graph older than a new generation, then add one
        os.utime(server.GRAPH_PATH, (0, 0))
        add_clip(root, "voiceB", "clip4", "Fourth sentence.")

        texts = sorted(item["text"] for item in server.generations("voiceB"))
        assert texts == ["Fourth sentence.", "Third sentence."], texts


def test_no_outputs_is_empty():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        point_server_at(root)
        assert server.list_voices() == []
        assert server.generations("voiceA") == []


if __name__ == "__main__":
    tests = [test_builds_graph_when_missing, test_rebuilds_when_a_new_clip_is_added, test_no_outputs_is_empty]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
