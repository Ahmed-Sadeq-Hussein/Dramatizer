"""main_backend.py | One place to run the back end: the code and the data.

    python main_backend.py                          # what can be run, and how
    python main_backend.py wer clone.wav "the words that should be heard"
    python main_backend.py split                    # make the voice manifest and its split (once)
    python main_backend.py test                     # every test that needs no GPU

Everything after the command's name goes to that command as it is, so each one
takes the same options as before. `python main_backend.py NAME --help` shows them,
for the commands that have options.

The back end lives in backend/:

    backend/model.py          every model, and the settings from .env
    backend/audio_io.py       reading a recording the way the measurements need it
    backend/metrics/          WER, similarity, quality, the rewards, the measurement log
    backend/voice_data/       the datasets: reading, picking, the manifest, the split
    backend/speech/           cloning a voice (CosyVoice 3), cleaning and combining references
    backend/genome/           genomes (delivery edits) and the search for good ones
    backend/emphasis/         word emphasis chosen by a small LLM, and its tests
    backend/evaluation/       checks of the manifest and the split, and run tracking
    backend/mcp_server/       the knowledge graph of the generated clips, for an assistant
    backend/rag/              the evidence corpus and its retrievers (course milestone 2)
    backend/data/             the committed data: manifests, the split, the emphasis test set

The front end (the website and logging in) is run by main_frontend.py.
"""
import runpy
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# name: (what it does, the module to run). A path instead of a module is a script
# that imports the files next to it, so it runs from its own folder.
COMMANDS = {
    "Measuring": {
        "wer": ("Word error rate of a recording against its script", "backend.metrics.WER"),
        "similarity": ("How alike two voices sound", "backend.metrics.Similarity"),
        "quality": ("Estimated quality of one recording (PESQ, STOI, SI-SDR)", "backend.metrics.PERQ"),
        "calibrate": ("Find the same/different speaker similarity limits", "backend.metrics.calibrate_similarity"),
        "calibrate-sanity": ("Measure how plain renders differ by chance, to set the sanity limits (renders)",
                             "backend.metrics.calibrate_sanity"),
    },
    "Voices and data": {
        "voices": ("Browse a dataset: list clips, hear one, read its script", "backend.voice_data.Voice_Importer"),
        "pick": ("Choose which clips of a dataset to use as references", "backend.voice_data.Picky_voice"),
        "manifest": ("Build the data manifest", "backend.voice_data.build_manifest"),
        "split": ("Make the voice manifest: each voice's longest clip and its validation/test split (once)", "backend.voice_data.split_voice_data"),
        "voice-similarity": ("Every voice against every other (Resemblyzer heatmap, GPU) and the closest voice with a "
                             "genome: build | heatmap | closest FILE.wav | check", "backend.voice_data.voice_similarity"),
        "voice-map": ("3D map and clusters of the validation voices by similarity; --pick N for a spread-out choice",
                      "backend.voice_data.voice_map"),
    },
    "Speech": {
        "speak": ("Speak a line in a cloned voice (CosyVoice 3)", "backend.speech.Voice_model"),
        "clean": ("Remove background noise from a recording", "backend.speech.Clean_Audio_Distortion"),
        "combine": ("Combine two recordings into one reference", "backend.speech.Combine_audio"),
        "format": ("Add paralinguistic tags to a line", "backend.speech.tts_formatter"),
    },
    "Genomes": {
        "genome": ("Find genomes for a voice with random or preference-guided search",
                   "backend.genome.search"),
        "preference": ("Train search guidance from rated runs or record an explicit listener choice",
                       "backend.genome.preference"),
        "ratings": ("Review listener scores (MOS) of genome searches; --named, --export FILE.csv",
                    "backend.genome.ratings"),
        "evidence-charts": ("Charts for the six genome questions, from scores given --from DATE --to DATE",
                            "backend.genome.evidence_charts"),
        "explore": ("Far-apart genomes (up to 10) for the voice-map voices, in 3 piles for MOS listeners (renders)",
                    "backend.genome.explore"),
    },
    "API (for others' PCs, through playit.gg)": {
        "api": ("Start the back end as a web API on https://127.0.0.1:5005; tunnel only this port",
                "backend.api.server"),
        "api-key": ("Who may use the API: api-key add NAME | list | remove NAME", "backend.api.keys"),
        "account": ("Login accounts: account add USERNAME --name NAME [--role listener|researcher|admin] | list | reset | disable | remove",
                    "backend.api.accounts"),
    },
    "Emphasis": {
        "emphasis-test": ("Which small LLM picks the right words to stress", "backend.emphasis.test_emphasis_llms"),
        "emphasis-plot": ("Graphs of the emphasis test", "backend.emphasis.plot_emphasis_results"),
    },
    "Evaluation": {
        "checks": ("Deterministic checks of the manifest, the split and the data contract", "backend/evaluation/checks.py"),
        "baseline": ("The reproducible baseline: plain renders of validation voices, measured and checked (renders)",
                     "backend/evaluation/baseline.py"),
        "compare-search": ("Milestone 2: random genome search on the baseline's voices and lines, against the baseline "
                           "(renders with --run)", "backend/evaluation/compare_search.py"),
        "sanity-all-genes": ("The sanity check of every gene type on its own, on baseline v2's voices and lines "
                             "(renders with --run)", "backend/evaluation/sanity_all_genes.py"),
        "eval-manifest": ("List every clip once, in a fixed order (history/eval)", "backend/evaluation/make_manifest.py"),
        "track": ("Record an evaluation run with MLflow", "backend/evaluation/track.py"),
        "import-mos": ("Log paired baseline and genome listening ratings to MLflow", "backend/evaluation/import_mos.py"),
        "graph": ("Build the knowledge graph of the generated clips", "backend/mcp_server/build_graph.py"),
        "mcp": ("Start the MCP server for the knowledge graph", "backend/mcp_server/dramatizer_mcp.py"),
    },
    "Evidence (questions answered from the project's records)": {
        "rag": ("rag corpus | label | refresh [--check] | ask \"a question\" | run: the evidence, its answer key, refresh, BM25 baseline",
                "backend.rag.cli"),
    },
}

# The tests that need no models or GPU, run as they are run by hand, with the
# package each needs beyond requirements.txt (None: nothing more). The MCP server
# has its own environment (see requirements.txt), so its test is skipped where
# "mcp" is not installed.
TESTS = [("backend/evaluation/test_checks.py", None), ("backend/evaluation/test_sanity_all_genes.py", None), ("backend/voice_data/test_voice_map.py", None), ("backend/voice_data/test_voice_similarity.py", None), ("backend/genome/test_explore.py", None), ("backend/genome/test_editor.py", None), ("backend/genome/test_evidence_charts.py", None), ("backend/api/test_accounts.py", None), ("backend/rag/test_rag.py", None), ("backend/mcp_server/test_graph.py", "mcp")]


def all_commands():
    return {name: target for group in COMMANDS.values() for name, (_, target) in group.items()}


def show_help():
    print(__doc__.split("\n\nThe back end lives")[0].split("\n", 1)[1])
    for group, commands in COMMANDS.items():
        print(f"\n{group}:")
        for name, (what, _) in commands.items():
            print(f"  {name:17s} {what}")
    print(f"\n  {'test':15s} Run every test that needs no GPU")


def run(target, args):
    """Run one command as if it were started on its own, with `args` as its command line."""
    sys.argv = [target, *args]
    if target.endswith(".py"):
        script = PROJECT_DIR / target
        sys.path.insert(0, str(script.parent))
        runpy.run_path(str(script), run_name="__main__")
    else:
        runpy.run_module(target, run_name="__main__", alter_sys=True)


def run_tests():
    import importlib.util

    failed, skipped = [], []
    for test, needs in TESTS:
        print(f"\n=== {test}", flush=True)
        if needs and importlib.util.find_spec(needs) is None:
            print(f"skipped: needs the package {needs!r}, which is not in this environment")
            skipped.append(test)
            continue
        if subprocess.run([sys.executable, str(PROJECT_DIR / test)], cwd=PROJECT_DIR).returncode:
            failed.append(test)
    print(("\nAll tests passed." if not failed else f"\nFailed: {', '.join(failed)}.")
          + (f" Skipped: {', '.join(skipped)}." if skipped else ""))
    return 1 if failed else 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help", "help"):
        show_help()
        return 0
    name, args = argv[0], argv[1:]
    if name == "test":
        return run_tests()
    commands = all_commands()
    if name not in commands:
        print(f"Unknown command {name!r}. Run `python main_backend.py` to see them all.", file=sys.stderr)
        return 2
    run(commands[name], args)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_DIR))
    sys.exit(main(sys.argv[1:]))
