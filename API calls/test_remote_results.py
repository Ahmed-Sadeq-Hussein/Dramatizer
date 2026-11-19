"""Tests of remote_results.py: the hosted copy answers like the API, and says clearly when it cannot.

    python "API calls/test_remote_results.py"
"""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import remote_results as R  # noqa: E402
from API_REQUEST import ApiError  # noqa: E402


def clip(listener, genome, score, above, types, when="2030-01-01 10:00:00"):
    return {"id": f"clip:r1/genome_{genome:02d}@{listener.lower()}", "kind": "clip", "revision": when,
            "title": f"VCTK/p250 genome {genome}, scored {score} by {listener}",
            "text": f"{listener} scored VCTK/p250 genome {genome} {score}, gene types {' '.join(types)}",
            "fields": {"voice": "VCTK/p250", "dataset": "VCTK", "run": "r1", "clip": f"genome_{genome:02d}",
                       "genome": genome, "listener": listener, "pile": 1, "score": score, "above_plain": above,
                       "features": len(types), "gene_types": types, "verdict": "fine", "artifact": None}}


def make_copy(folder, records):
    folder = Path(folder)
    snapshot = {"provenance": {"snapshot_id": "abc123", "ratings_up_to": "2030-01-02 09:00:00", "runs": ["r1"],
                               "code_commit": "x", "rule": "r", "winner_rule": "w"},
                "coverage": {"listeners_per_genome": {"1": 2}}, "caveats": [], "voices": [], "genomes": []}
    files = {"evidence_records.jsonl": "".join(json.dumps(r) + "\n" for r in records),
             "genomes.json": json.dumps({"r1": {"1": [["sigh after 'well'"], "Well [sigh]"], "2": [["speed 1.2"], "Well"]}}),
             "snapshot.json": json.dumps(snapshot), "results.json": json.dumps({"baselines": [], "milestone2": []})}
    for name, text in files.items():
        (folder / name).write_bytes(text.encode("utf-8"))           # as published: no Windows line endings
    manifest = {"snapshot_id": "abc123", "files": {n: {"sha256": hashlib.sha256(t.encode()).hexdigest()}
                                                   for n, t in files.items()}}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class Remote(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        make_copy(self.tmp.name, [clip("Julius", 1, 4.0, 1.0, ["sigh"]), clip("Rama", 2, 0.0, -3.0, ["speed"],
                                                                              when="2030-01-03 08:00:00")])
        self.results = R.connect(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_scores_and_answers_come_from_the_copy_with_its_data_state(self):
        rows = self.results.query_scores()["rows"]
        self.assertEqual([(r["listener"], r["score"]) for r in rows], [("Julius", 4.0), ("Rama", 0.0)])
        self.assertEqual(rows[0]["genes_described"], ["sigh after 'well'"])
        found = self.results.search_evidence("sigh")
        self.assertTrue(found["hits"])
        self.assertEqual(found["provenance"], {"snapshot_id": "abc123", "ratings_up_to": "2030-01-02 09:00:00",
                                               "data_revision": "2030-01-03 08:00:00", "scope": "every listener"})
        self.assertEqual(self.results.baseline_results(), {"baselines": []})

    def test_what_only_the_live_api_knows_is_refused_with_the_reason(self):
        answer = json.loads(self.results.run_tool("top_genomes", {}))
        self.assertIn("live API only", answer["error"])
        with self.assertRaises(ApiError):
            self.results.evidence_question(1, from_date="2030-01-01")

    def test_a_changed_file_is_caught_by_the_manifest(self):
        (Path(self.tmp.name) / "results.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ApiError, "does not match the manifest"):
            R.connect(self.tmp.name).baseline_results()

    def test_a_login_page_instead_of_data_says_how_to_log_in(self):
        site = R.RemoteSite(self.tmp.name)
        site._files["manifest"] = {"files": {}}
        site._files["results.json"] = b"<!doctype html><title>Sign in</title>"
        with self.assertRaisesRegex(ApiError, "DRAMATIZER_ACCESS_ID"):
            site.read_json("results.json")


class HuggingFace(unittest.TestCase):
    def test_a_private_dataset_is_read_with_the_readers_own_token(self):
        site = R.RemoteSite("hf://datasets/team/dramatizer-results@abc123", hf_token="hf_x")
        self.assertEqual(site.where, "https://huggingface.co/datasets/team/dramatizer-results/resolve/abc123")
        self.assertEqual(site.headers["Authorization"], "Bearer hf_x")
        self.assertTrue(R.RemoteSite("hf://team/dramatizer-results", hf_token="t").where.endswith("/resolve/main"))
        with self.assertRaises(ApiError):
            R.RemoteSite("hf://datasets/no-owner")


if __name__ == "__main__":
    unittest.main()
