"""Tests of the evidence corpus and the BM25 baseline (no models, no real ratings).

    python -m unittest backend.rag.test_rag
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))    # run as a file by main_backend.py test

from backend.rag import baseline, corpus  # noqa: E402


def record(id_, text, kind="clip", revision="2026-11-11 10:00:00"):
    return {"id": id_, "kind": kind, "revision": revision, "title": id_, "text": text, "fields": {}}


def run(voice="CREMA-D/1014", run_id="mos-diverse-v1_p1_CREMA-D_1014"):
    genes = [{"feature": "vocalized_noise", "position": 0.2}, {"feature": "pause", "position": 0.5, "kind": "dramatic"}]
    return {"id": run_id, "batch": "mos-diverse-v1", "pile": 1, "voice": voice, "text": "Well, you really did it.",
            "genomes": [{"number": 1, "genes": genes, "described": ["vocalized noise after 'Well'", "dramatic pause after 'did'"],
                         "speed": 1.0, "verdict": "fine", "markup": "Well [vocalized_noise] ..."},
                        {"number": 2, "genes": genes[:1], "described": ["vocalized noise after 'Well'"],
                         "speed": 1.0, "verdict": "fine", "markup": "Well [vocalized_noise]"}]}


def row(clip, genome, score, above, rated_on="2026-11-11 10:00:00", artifact=""):
    return {"rater": "Ash", "account": "ash", "batch": "mos-diverse-v1", "pile": "1", "voice": "CREMA-D/1014",
            "run": "mos-diverse-v1_p1_CREMA-D_1014", "clip": clip, "genome": genome, "score": score, "artifact": artifact,
            "above_plain_line": above, "rated_on": rated_on, "first_given": rated_on, "changes": "0", "verdict": "fine",
            "markup": "", "text": ""}


class Words(unittest.TestCase):
    def test_split_lower_and_voice_names(self):
        got = baseline.words("Which genome of CREMA-D/1014 has vocalized_noise?")
        self.assertIn("vocalized", got)
        self.assertIn("noise", got)
        self.assertIn("cremad1014", got)        # the voice as one word
        self.assertNotIn("which", got)          # a stop word

    def test_vctk_voice(self):
        self.assertIn("vctkp250", baseline.words("voice VCTK/p250"))


class Bm25(unittest.TestCase):
    def setUp(self):
        self.index = baseline.BM25([
            record("clip:a/genome_01", "Voice CREMA-D/1014 genome 1 has laughter and a sigh."),
            record("clip:a/genome_02", "Voice CREMA-D/1014 genome 2 has hissing."),
            record("clip:b/genome_01", "Voice VCTK/p250 genome 1 has hissing and hissing again."),
            record("sanity:x", "Sanity check failed: words dropped.", kind="sanity"),
        ])

    def test_the_record_with_the_rare_words_ranks_first(self):
        hits = self.index.search("Which genome of CREMA-D/1014 has hissing?")
        self.assertEqual(hits[0]["id"], "clip:a/genome_02")

    def test_hits_carry_id_kind_revision_and_rank(self):
        hit = self.index.search("sanity failed")[0]
        self.assertEqual((hit["rank"], hit["id"], hit["kind"], hit["revision"]), (1, "sanity:x", "sanity", "2026-11-11 10:00:00"))

    def test_kind_filter_and_k(self):
        self.assertEqual([h["kind"] for h in self.index.search("genome", kind="clip")], ["clip"] * 3)
        self.assertEqual(len(self.index.search("genome", k=2)), 2)

    def test_no_shared_word_finds_nothing(self):
        self.assertEqual(self.index.search("angry pirate"), [])

    def test_same_question_same_ranking(self):
        self.assertEqual(self.index.search("genome hissing"), self.index.search("genome hissing"))


class Corpus(unittest.TestCase):
    def test_clip_and_run_records(self):
        runs = {"mos-diverse-v1_p1_CREMA-D_1014": run()}
        rows = [row("baseline", "", "3.0", ""), row("genome_01", "1", "4.5", "1.5", artifact="yes"),
                row("genome_02", "2", "2.0", "-1.0", rated_on="2026-11-11 10:05:00")]
        clips = corpus.clip_records(rows, runs)
        self.assertEqual([c["id"] for c in clips], ["clip:mos-diverse-v1_p1_CREMA-D_1014/baseline",
                                                    "clip:mos-diverse-v1_p1_CREMA-D_1014/genome_01",
                                                    "clip:mos-diverse-v1_p1_CREMA-D_1014/genome_02"])
        one = clips[1]
        self.assertEqual(one["revision"], "2026-11-11 10:00:00")
        self.assertIn("better than the plain line", one["text"])
        self.assertIn("pause (dramatic)", one["text"])
        self.assertEqual((one["fields"]["features"], one["fields"]["artifact"]), (2, True))
        self.assertIn("worse than the plain line", clips[2]["text"])

        runs_out = corpus.run_records(clips, runs)
        self.assertEqual(runs_out[0]["id"], "run:mos-diverse-v1_p1_CREMA-D_1014")
        self.assertEqual(runs_out[0]["revision"], "2026-11-11 10:05:00")      # the newest score in the run
        self.assertEqual(runs_out[0]["fields"]["best_genomes"], [1])
        self.assertEqual(runs_out[0]["fields"]["better_than_plain"], 1)

    def test_sanity_records_and_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "record.json").write_text(json.dumps({"name": "random-v2", "made": "2026-11-08 03:13:11"}), encoding="utf-8")
            (folder / "rows.json").write_text(json.dumps([{
                "voice": "VCTK_p231", "dataset": "VCTK", "line": 0, "genome": 2, "verdict": "terrible",
                "features": ["clucking", "noise", "sigh"], "markup": "x", "wer_ok": False, "similarity_ok": True,
                "pace_ok": True, "wer": 0.4, "similarity": 0.91, "heard": "y"}]), encoding="utf-8")
            records = corpus.sanity_records(folder)
            self.assertEqual(records[0]["id"], "sanity:random-v2/VCTK_p231/line0/genome02")
            self.assertEqual(records[0]["revision"], "2026-11-08 03:13:11")
            self.assertIn("words dropped or changed", records[0]["text"])

            manifest = corpus.write(records, folder / "corpus.jsonl", sources=[folder / "rows.json"])
            self.assertEqual((manifest["records"], manifest["kinds"]), (1, {"sanity": 1}))
            self.assertEqual(corpus.load(folder / "corpus.jsonl")[0]["id"], records[0]["id"])
            self.assertTrue((folder / "corpus_manifest.json").is_file())


def clip_record(run_id, clip, genome, score, above, voice="VCTK/p250", types=(), artifact=None, revision="2026-11-11 10:00:00"):
    return {"id": f"clip:{run_id}/{clip}", "kind": "clip", "revision": revision, "title": clip, "text": f"{clip} {score}",
            "fields": {"voice": voice, "run": run_id, "clip": clip, "genome": genome, "score": score,
                       "above_plain": above, "gene_types": list(types), "artifact": artifact}}


def small_corpus():
    r = "mos-diverse-v1_p1_VCTK_p250"
    return [clip_record(r, "baseline", None, 3.0, None),
            clip_record(r, "genome_01", 1, 4.0, 1.0, types=["sigh"]),
            clip_record(r, "genome_02", 2, 4.0, 1.0, types=["speed"]),
            clip_record(r, "genome_03", 3, 1.0, -2.0, types=["hissing"], artifact=True),
            {"id": f"run:{r}", "kind": "run", "revision": "2026-11-11 10:00:00", "title": "", "text": "",
             "fields": {"voice": "VCTK/p250", "run": r, "better_than_plain": 2, "best_genomes": [1, 2]}},
            {"id": "sanity:random-v2/a", "kind": "sanity", "revision": "2026-11-08 03:13:11", "title": "", "text": "",
             "fields": {"verdict": "failed", "gene_types": ["speed", "sigh"]}},
            {"id": "sanity:random-v2/b", "kind": "sanity", "revision": "2026-11-08 03:13:11", "title": "", "text": "",
             "fields": {"verdict": "passed", "gene_types": ["mn"]}}]


class Labels(unittest.TestCase):
    def test_rules(self):
        from backend.rag import labels
        rs, r = small_corpus(), "mos-diverse-v1_p1_VCTK_p250"
        self.assertEqual(labels.best_per_voice(rs), {f"run:{r}": 2, f"clip:{r}/genome_01": 1, f"clip:{r}/genome_02": 1})
        self.assertEqual(labels.bad_mos(rs), {f"clip:{r}/genome_03": 2})
        self.assertEqual(labels.artifacts(rs), {f"clip:{r}/genome_03": 2, "sanity:random-v2/a": 1})
        self.assertEqual(labels.better_than_plain(rs), {f"clip:{r}/genome_01": 2, f"clip:{r}/genome_02": 2, f"run:{r}": 1})
        self.assertEqual(labels.exclude(rs), {f"clip:{r}/genome_02": 1, f"clip:{r}/genome_03": 2, "sanity:random-v2/a": 1})
        self.assertEqual(labels.voice_better("VCTK/p250")(rs)[f"clip:{r}/baseline"], 1)
        self.assertEqual(labels.nothing(rs), {})

    def test_label_keeps_revisions_and_what_to_expect(self):
        from backend.rag import labels
        questions = [{"id": "q03", "question": "bad?"}, {"id": "q09", "question": "angry?"}]
        got = labels.label(small_corpus(), questions)
        self.assertEqual(got["q03"]["expect"], "answer")
        self.assertEqual(list(got["q03"]["relevant"].values()), [{"grade": 2, "revision": "2026-11-11 10:00:00"}])
        self.assertEqual((got["q09"]["expect"], got["q09"]["relevant"]), ("decline", {}))
        self.assertIn("worse than the plain line", got["q03"]["rule"])


class Refresh(unittest.TestCase):
    def test_compare_finds_added_removed_and_changed(self):
        from backend.rag import refresh
        old = small_corpus()
        new = small_corpus()[1:]                                         # the plain line is gone
        new[0] = {**new[0], "revision": "2026-11-12 09:00:00"}          # genome 1 scored again
        new.append(clip_record("x", "genome_01", 1, 2.0, 0.0))
        diff = refresh.compare(old, new)
        self.assertEqual(diff["removed"], ["clip:mos-diverse-v1_p1_VCTK_p250/baseline"])
        self.assertEqual(diff["changed"], ["clip:mos-diverse-v1_p1_VCTK_p250/genome_01"])
        self.assertEqual(diff["added"], ["clip:x/genome_01"])

    def test_stale_labels(self):
        from backend.rag import labels, refresh
        questions = [{"id": "q03", "question": "bad?"}, {"id": "q04", "question": "better?"}]
        saved = {"questions": labels.label(small_corpus(), questions)}
        self.assertEqual(refresh.stale_labels(saved, small_corpus(), questions)[0], {})     # nothing moved
        later = small_corpus()
        later[1] = clip_record("mos-diverse-v1_p1_VCTK_p250", "genome_01", 1, 2.0, -1.0, revision="2026-11-12 09:00:00")
        stale, _ = refresh.stale_labels(saved, later, questions)
        self.assertEqual(stale["q03"]["added"], ["clip:mos-diverse-v1_p1_VCTK_p250/genome_01"])     # now worse than plain
        self.assertEqual(stale["q04"]["dropped"], ["clip:mos-diverse-v1_p1_VCTK_p250/genome_01"])
        self.assertEqual(stale["q04"]["revised"], ["clip:mos-diverse-v1_p1_VCTK_p250/genome_01"])

    def test_check_sees_a_changed_source(self):
        from backend.rag import refresh
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / "who.csv"
            source.write_text("a", encoding="utf-8")
            manifest = corpus.write([record("clip:x", "x")], folder / "corpus.jsonl", sources=[source])
            self.assertTrue(manifest["sources"])
            state = refresh.check(folder / "corpus_manifest.json", [source])
            self.assertEqual((state["changed"], state["missing"]), ([], []))
            source.write_text("b", encoding="utf-8")
            self.assertEqual(refresh.check(folder / "corpus_manifest.json", [source])["changed"], [str(source)])
            self.assertEqual(refresh.check(folder / "corpus_manifest.json", [folder / "gone.csv"])["missing"],
                             [str(folder / "gone.csv")])


if __name__ == "__main__":
    unittest.main()
