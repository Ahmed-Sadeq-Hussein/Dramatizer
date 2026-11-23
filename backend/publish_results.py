"""publish_results.py | The results, without audio, as a folder of static files a host can serve while this PC is off.

    python main_backend.py publish-results              # build the bundle in OUTPUT_DIR/results_site and check it
    python main_backend.py publish-results --zip        # also a zip of it, for Cloudflare's dashboard upload
    python main_backend.py publish-results --deploy     # also upload it with wrangler (needs Node.js and a login)
    python main_backend.py publish-results --hf OWNER/NAME   # or upload it to a private Hugging Face dataset

The bundle is read-only and meant for the team only: host it behind a login (Cloudflare Access, or a private
Hugging Face dataset, which --hf checks before uploading). It holds
every listener's scores, so it is for researchers and admins, not for listeners who rate blind.

    index.html              a page with the stamps, the evidence snapshot's summary and the six charts
    manifest.json           when it was made, what it was made from (snapshot_id, data_revision,
                            ratings_up_to, code commit), every file with its size and sha256
    snapshot.json           the evidence snapshot (GET /evidence/snapshot): coverage, winners, checks, caveats
    scores.csv              every listener's current score of every clip (who_rated_what.csv), without the
                            login names
    evidence_records.jsonl  the records answers and search are made from (GET /evidence/answer, /search)
    rating_history.csv      every score as it was given or changed, oldest first (the rating log), with the
                            score before it; without login names
    runs.json               every run that is not sealed, rated or not: voice, batch, pile, method, seeds, the
                            line and reference text, the plain line, and every genome in full (genes,
                            operations, markup, speed, warnings, verdict, sanity: word error rate, speaker
                            similarity, what Whisper heard, pace); who rated it. Not the audio file names
    piles.json              which listener was given which pile, by display name (sealed piles left out)
    genomes.json            what each scored genome does and its markup (GET /evidence/scores)
    charts/                 the six questions' charts (JPEG) and charts.json with their claims and notes
    mos_study/              the listening studies from ASSIGNED piles only (mos_study.py): summary.json (the
                            Report evaluation's three methods compared, totals only, and the pilot), its charts,
                            and scores_assigned.csv (the pilot's assigned scores, without login names)
    results.json            the baselines' and Milestone 2's summaries (GET /results)

Never in it, and checked before anything is written out (check() fails the build otherwise): sealed runs
(the Report evaluation until its key is opened), key.json, scores of test accounts, accounts, emails,
passwords, API keys and local paths. Audio is never copied.
"""
import argparse
import csv
import hashlib
import html
import io
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from backend import model

OUT_DIR = model.OUTPUT_DIR / "results_site"
PROJECT = "dramatizer-results"                       # the Cloudflare Pages project name
DROPPED_COLUMNS = ("account",)                       # login names stay on this PC
FORBIDDEN = [
    (re.compile(r"key\.json"), "the sealed key file"),
    (re.compile(r"report-evaluation"), "the sealed Report evaluation"),
    (re.compile(r"test_user_\d+"), "a test account"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+\.?[a-z]{2,}", re.IGNORECASE), "an email address"),
    (re.compile(r"password|api[_-]?key|bearer\s", re.IGNORECASE), "a credential word"),
    (re.compile(r"[A-Za-z]:[\\/]+(Users|envs|Dramatizer)", re.IGNORECASE), "a local path"),
]
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".html", ".txt"}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def sealed_runs():
    from backend.genome import search
    return {run["id"] for _, run in search.list_runs() if run.get("sealed")}


def scores_csv(sealed):
    """who_rated_what.csv without test accounts, sealed runs or login names: (text, rows kept)."""
    from backend.genome import ratings
    from backend.rag import corpus

    path = ratings.who_path()
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = [c for c in reader.fieldnames if c not in DROPPED_COLUMNS]
        rows = [{k: row[k] for k in fields} for row in reader
                if row["run"] not in sealed and not corpus.is_test_account(row)]
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue(), len(rows)


def results_summaries():
    """The baselines' and Milestone 2's summaries, as GET /results gives them."""
    from backend.api import server
    return server.results_data()


def genome_texts(records):
    """{run: {genome number: [described, markup]}} for the runs the evidence records score (as the API has it)."""
    from backend.genome import search

    runs = {r["fields"]["run"] for r in records if r["kind"] == "clip"}
    return {run["id"]: {str(gm["number"]): [gm.get("described") or [], gm.get("markup")] for gm in run["genomes"]}
            for _, run in search.list_runs() if run["id"] in runs and not run.get("sealed")}


RUN_LEFT_OUT = ("reference_audio", "only")          # a local path to the reference clip; login names
AUDIO_FIELDS = ("audio",)                            # file names of the clips: the audio itself stays on this PC


def runs_data(runs, rated, sealed):
    """{run id: the run as stored, without audio file names, local paths or login names} for every run that is
    not sealed: its settings, the plain line (what Whisper heard, its length), and every genome with its genes,
    operations, markup, speed, warnings, verdict and full sanity measurement. `rated`: {run: rating_runs row},
    which adds who rated it, the picked genomes and when it was last scored."""
    out = {}
    for run in runs:
        if run.get("sealed") or run["id"] in sealed:
            continue
        one = {k: v for k, v in run.items() if k not in RUN_LEFT_OUT + ("genomes", "baseline")}
        one["baseline"] = {k: v for k, v in (run.get("baseline") or {}).items() if k not in AUDIO_FIELDS}
        one["genomes"] = [{k: v for k, v in g.items() if k not in AUDIO_FIELDS} for g in run.get("genomes", [])]
        head = rated.get(run["id"])
        one["ratings"] = ({"raters": [r for r in json.loads(head["raters"]) if not _is_test(r)],
                           "picked": json.loads(head["picked"]), "updated": head["updated"]} if head else None)
        out[run["id"]] = _strip_paths(one)
    return out


HISTORY_FIELDS = ["time", "run", "batch", "pile", "voice", "rater", "submitted_by", "clip", "genome", "score",
                  "previous", "artifact", "previous_artifact", "different", "previous_different", "speed", "verdict",
                  "genes", "markup", "text"]


def _is_test(name):
    from backend.genome import ratings
    return ratings.is_test_rater(name) or bool(re.match(r"^test_user_\d+$", str(name or "")))


def history_csv(entries, sealed):
    """The rating log (every score as it was given or changed, oldest first) without login names, test accounts'
    scores or sealed runs: (text, rows kept)."""
    rows = []
    for e in entries:
        if e.get("run") in sealed or _is_test(e.get("rater")) or _is_test(e.get("account")):
            continue
        row = {k: e.get(k) for k in HISTORY_FIELDS}
        row["genes"] = json.dumps(e.get("genes") or [], ensure_ascii=False)
        if _is_test(row["submitted_by"]):
            row["submitted_by"] = None
        rows.append(row)
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=HISTORY_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue(), len(rows)


def piles_data(assignments, accounts, sealed_batches):
    """{"listeners": [{name, role, piles}]}: who was given which pile to rate, by display name. Test accounts and
    the sealed batches' piles are left out (a listener with only sealed piles is listed with none)."""
    listeners = []
    for account in accounts:
        if _is_test(account["username"]) or _is_test(account["display_name"]):
            continue
        piles = [a["pile"] for a in sorted(assignments, key=lambda a: a["position"])
                 if a["username"] == account["username"] and a["pile"].split("|")[0] not in sealed_batches]
        listeners.append({"name": account["display_name"], "role": account["role"],
                          "piles": [{"batch": p.split("|")[0], "pile": int(p.split("|")[1])} for p in piles]})
    return {"listeners": listeners}


def rating_tables():
    """(runs, rating_runs rows by run, rating log entries, pile assignments, accounts) from this PC's database."""
    from backend import storage
    from backend.genome import search

    with storage.connect() as db:
        rated = {r["run"]: dict(r) for r in db.execute("SELECT run, raters, picked, updated FROM rating_runs")}
        log = [json.loads(r[0]) for r in db.execute("SELECT entry FROM rating_log ORDER BY id")]
        assignments = [dict(r) for r in db.execute("SELECT username, pile, position FROM account_piles")]
        accounts = [dict(r) for r in db.execute("SELECT username, display_name, role FROM accounts")]
    return [run for _, run in search.list_runs()], rated, log, assignments, accounts


def commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=model.PROJECT_DIR, capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def check(folder, sealed):
    """Every forbidden thing found in the bundle's text files: [(file, what)]. Empty means it may be published."""
    found = []
    patterns = FORBIDDEN + [(re.compile(re.escape(r)), f"the sealed run {r}") for r in sorted(sealed)]
    for path in sorted(Path(folder).rglob("*")):
        if path.is_file() and path.suffix.lower() in (".wav", ".flac", ".mp3", ".npz", ".zip"):
            found.append((path.name, "an audio or archive file"))
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern, what in patterns:
            if pattern.search(text):
                found.append((str(path.relative_to(folder)), what))
    return found


def _strip_paths(value):
    """Charts' summaries name their files by full path: keep only the file name."""
    if isinstance(value, dict):
        return {k: _strip_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_paths(v) for v in value]
    if isinstance(value, str) and re.match(r"^[A-Za-z]:[\\/]", value):
        return Path(value).name
    return value


def build(out_dir=OUT_DIR):
    """Make the bundle in a fresh folder next to out_dir, check it, then swap it in. Returns the manifest."""
    from backend.genome import evidence_charts
    from backend.genome import evidence_snapshot as E
    from backend.rag import corpus

    out_dir = Path(out_dir)
    staging = out_dir.with_name(out_dir.name + ".new")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "charts").mkdir(parents=True)

    sealed = sealed_runs()
    snapshot = E.build()
    records = corpus.build()
    data_revision = max((r["revision"] for r in records if r["revision"]), default=None)
    scores, kept = scores_csv(sealed)
    charts = evidence_charts.make_charts(records=records, out_dir=staging / "charts")
    (staging / "charts" / "summary.json").unlink(missing_ok=True)   # full paths; charts.json replaces it

    def write(name, text):
        (staging / name).write_bytes(text.encode("utf-8"))      # the same bytes on every machine (no \r\n)

    write("snapshot.json", json.dumps(snapshot, ensure_ascii=False, indent=1))
    write("scores.csv", scores)
    write("evidence_records.jsonl", "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records))
    write("charts/charts.json", json.dumps(_strip_paths(charts), ensure_ascii=False, indent=1))
    write("genomes.json", json.dumps(genome_texts(records), ensure_ascii=False, indent=1))
    write("results.json", json.dumps(_strip_paths(results_summaries()), ensure_ascii=False, indent=1))

    study = mos_study_files(staging / "mos_study")

    all_runs, rated, log, assignments, accounts = rating_tables()
    sealed_batches = {run.get("batch") for run in all_runs if run.get("sealed") and run.get("batch")}
    runs = runs_data(all_runs, rated, sealed)
    history, changes = history_csv(log, sealed)
    write("runs.json", json.dumps(runs, ensure_ascii=False, indent=1))
    write("rating_history.csv", history)
    write("piles.json", json.dumps(piles_data(assignments, accounts, sealed_batches), ensure_ascii=False, indent=1))

    provenance = snapshot["provenance"]
    stamps = {"made": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "snapshot_id": provenance["snapshot_id"],
              "ratings_up_to": provenance["ratings_up_to"], "data_revision": data_revision, "code_commit": commit(),
              "scores": kept, "rating_history": changes, "runs": len(runs),
              "rated_runs": sum(1 for r in runs.values() if r["ratings"]), "evidence_records": len(records),
              "sealed_runs_left_out": len(sealed)}
    write("index.html", page(stamps, snapshot, charts, study))
    files = {str(p.relative_to(staging)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": _sha(p.read_bytes())}
             for p in sorted(staging.rglob("*")) if p.is_file()}
    manifest = {**stamps, "about": "Dramatizer results without audio, for the team only. Never holds sealed runs, "
                                   "the condition key, accounts or test accounts' scores.", "files": files}
    write("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1))

    problems = check(staging, sealed)
    if problems:
        raise SystemExit("Not published, the bundle holds what it must not:\n"
                         + "\n".join(f"  {name}: {what}" for name, what in problems)
                         + f"\nThe draft is left in {staging} to look at.")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    staging.rename(out_dir)
    return manifest


def mos_study_files(folder):
    """The listening studies (assigned piles only) into folder: summary.json, charts, scores_assigned.csv."""
    from backend.genome import mos_study
    return mos_study.build(folder)


def study_html(study):
    """The Report evaluation's methods compared, for index.html (totals only)."""
    e = lambda x: html.escape(str(x))
    rep = study and study.get("report")
    if not rep:
        return '<p class="muted">The Report evaluation is not complete yet.</p>'
    rows = "".join(f"<tr><td>{e(c['label'])}</td><td>{c['mean']:.2f}</td><td>{c['ci95'][0]:.2f} to {c['ci95'][1]:.2f}</td>"
                   f"<td>{round(100 * c['share_4_or_more'])} %</td><td>{c['artifact_flags']}</td></tr>"
                   for c in rep["conditions"].values())
    pairs = "".join(f"<tr><td>{e(p['a'])} vs {e(p['b'])}</td><td>{p['mean_difference']:+.2f}</td>"
                    f"<td>{p['ci95'][0]:+.2f} to {p['ci95'][1]:+.2f}</td><td>{p['a_higher']} / {p['tie']} / {p['b_higher']}</td>"
                    f"<td>{e(p['holm_p'])}</td></tr>" for p in rep["paired"])
    h = rep.get("harm") or {}
    types = "".join(f"<tr><td>{e(t['gene_type'])}</td><td>{t['genomes']}</td><td>{t['minus_plain']:+.2f}</td>"
                    f"<td>{t['worse']} / {t['better']}</td><td>{'yes' if t['in_guarded'] else 'no'}</td></tr>"
                    for t in rep.get("gene_types", []))
    where = "".join(f"<tr><td>{e(k)}</td><td>{v['genomes']}</td><td>{v['minus_plain']:+.2f}</td><td>{v['worse']} / {v['better']}</td></tr>"
                    for k, v in {**h.get("positions", {}), **{f"{n} gene(s)": v for n, v in h.get("gene_count", {}).items()}}.items())
    harm_html = (f"<h3>Which genomes hurt (exploratory)</h3><p>The worst 10 of {h.get('genomes')} genomes carry "
                 f"{round(100 * (h.get('share_of_loss_in_worst_10') or 0))} % of the loss against the plain line; without the worst "
                 f"15 the rest average {(h.get('rest_without_worst') or {}).get('15', 0):+.2f}.</p>"
                 f"<table><tr><td><b>Gene type</b></td><td><b>Genomes</b></td><td><b>vs plain</b></td><td><b>Worse / better</b></td>"
                 f"<td><b>Allowed in guarded</b></td></tr>{types}</table><p></p>"
                 f"<table><tr><td><b>Where the events sit / genes</b></td><td><b>Genomes</b></td><td><b>vs plain</b></td>"
                 f"<td><b>Worse / better</b></td></tr>{where}</table>") if h else ""
    figs = "".join(f'<figure><img src="mos_study/{e(c["file"])}" alt="{e(c["title"])}"><figcaption><b>{e(c["title"])}</b>'
                   f'<br><span class="muted">{e(c["note"])}</span></figcaption></figure>' for c in study.get("charts", []))
    return (f"<p>{rep['scores']} scores of {rep['items']} items on {rep['voices']} test voices, from the piles each "
            f"listener was assigned only ({', '.join(e(x) for x in rep['listeners'])}).</p>"
            f"<table><tr><td><b>Method</b></td><td><b>Mean</b></td><td><b>95 % interval</b></td><td><b>Scored 4+</b></td>"
            f"<td><b>Artifact flags</b></td></tr>{rows}</table>"
            f"<p></p><table><tr><td><b>Pair</b></td><td><b>Mean difference</b></td><td><b>95 % interval</b></td>"
            f"<td><b>First higher / tie / second higher</b></td><td><b>p (Holm)</b></td></tr>{pairs}</table>"
            f"{harm_html}<div class=\"grid\">{figs}</div>")


def page(stamps, snapshot, charts, study=None):
    """index.html: one self-contained page (no scripts), light and dark."""
    e = lambda x: html.escape(str(x))
    status = {}
    for v in snapshot.get("voices", []):
        status[v["status"]] = status.get(v["status"], 0) + 1
    cov = snapshot.get("coverage", {})
    per_genome = ", ".join(f"{n} heard by {k} listener{'s' if k != '1' else ''}"
                           for k, n in sorted((cov.get("listeners_per_genome") or {}).items()))
    chart_html = []
    for c in charts["charts"]:
        if "file" in c:
            chart_html.append(f'<figure><img src="charts/{e(Path(c["file"]).name)}" alt="{e(c["question"])}">'
                              f'<figcaption><b>{e(c["number"])}. {e(c["question"])}</b><br>{e(c["claim"])}'
                              f'<br><span class="muted">{e(c["note"])}</span></figcaption></figure>')
        else:
            chart_html.append(f'<figure><figcaption><b>{e(c["number"])}. {e(c["question"])}</b><br>'
                              f'<span class="muted">No chart: {e(c["skipped"])}</span></figcaption></figure>')
    rows = "".join(f"<tr><td>{e(k)}</td><td>{e(v)}</td></tr>" for k, v in stamps.items())
    voices = "".join(f"<li>{e(k)}: {e(v)}</li>" for k, v in sorted(status.items()))
    caveats = "".join(f"<li>{e(c)}</li>" for c in snapshot.get("caveats", []))
    links = "".join(f'<li><a href="{f}">{f}</a></li>' for f in
                    ("manifest.json", "snapshot.json", "scores.csv", "rating_history.csv", "runs.json", "piles.json",
                     "genomes.json", "evidence_records.jsonl", "charts/charts.json", "results.json",
                     "mos_study/summary.json", "mos_study/scores_assigned.csv"))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dramatizer results</title>
<style>
:root {{ --bg:#fbfbfa; --fg:#1d1d1b; --muted:#6b6b66; --line:#e2e1dc; --card:#ffffff; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161615; --fg:#ecebe6; --muted:#a3a29b; --line:#33332f; --card:#1f1f1d; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,sans-serif; margin:0 auto; max-width:1100px; padding:24px 16px; }}
h1 {{ font-size:1.6rem; margin:0 0 4px; }} h2 {{ font-size:1.15rem; margin-top:32px; }}
.muted {{ color:var(--muted); }} table {{ border-collapse:collapse; }} td {{ border-bottom:1px solid var(--line); padding:4px 16px 4px 0; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:16px; }}
figure {{ background:var(--card); border:1px solid var(--line); border-radius:8px; margin:0; padding:12px; }}
img {{ width:100%; height:auto; border-radius:4px; }} figcaption {{ margin-top:8px; font-size:14px; }}
a {{ color:inherit; }}
</style></head><body>
<h1>Dramatizer results</h1>
<p class="muted">Read-only copy for the team. It is as new as the last publish; the stamps below say which data it holds.
The Report evaluation's runs stay sealed: only its totals per method are here.</p>
<h2>Stamps</h2><table>{rows}</table>
<h2>Evidence snapshot</h2>
<p>{e(cov.get("genomes_rated", "?"))} rated genomes; {e(per_genome)}.</p>
<p>Voices by status:</p><ul>{voices}</ul>
<p>Caveats:</p><ul>{caveats}</ul>
<h2>Listening study: the three methods (assigned piles only)</h2>{study_html(study)}
<h2>The six questions</h2><div class="grid">{''.join(chart_html)}</div>
<h2>Files</h2><ul>{links}</ul>
</body></html>
"""


def deploy(folder):
    """Upload with Cloudflare's wrangler (Node.js); the first run asks you to log in to Cloudflare in the browser."""
    npx = shutil.which("npx")
    if not npx:
        raise SystemExit("wrangler needs Node.js (npx), which is not installed. Upload the zip instead "
                         "(--zip, then Cloudflare dashboard > Workers & Pages > dramatizer-results > Create deployment).")
    return subprocess.run([npx, "wrangler", "pages", "deploy", str(folder), "--project-name", PROJECT,
                           "--branch", "main"]).returncode


BUNDLE_PATTERNS = ["*.json", "*.jsonl", "*.csv", "*.html", "charts/*"]     # what a new publish replaces


def hf_upload(folder, repo_id, manifest, token=None):
    """Upload the bundle to a PRIVATE Hugging Face dataset as one commit; refuses a public or missing one.
    Files of the previous publish that are no longer in the bundle are removed in the same commit.
    token: a write token (default: HF_TOKEN, or the login saved by `hf auth login`). Returns the commit."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

    api = HfApi(token=token)
    try:
        info = api.repo_info(repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        raise SystemExit(f"No dataset {repo_id} that this token can see. Create it as Private first "
                         f"(huggingface.co/new-dataset), and log in with a write token.") from None
    except HfHubHTTPError as error:
        raise SystemExit(f"Hugging Face refused: {error}") from None
    if not info.private:
        raise SystemExit(f"Not uploaded: {repo_id} is PUBLIC. The results hold every listener's scores; make the "
                         f"dataset private (its Settings > Change visibility) and run this again.")
    message = (f"Results {manifest['snapshot_id']}: ratings up to {manifest['ratings_up_to']}, "
               f"data revision {manifest['data_revision']}")
    commit = api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="dataset",
                               commit_message=message, delete_patterns=BUNDLE_PATTERNS)
    return commit


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the results bundle (no audio) for team-only hosting.")
    parser.add_argument("--out", default=str(OUT_DIR), help=f"the folder to build (default {OUT_DIR})")
    parser.add_argument("--zip", action="store_true", help="also write a zip next to it, for the dashboard upload")
    parser.add_argument("--deploy", action="store_true", help="upload with wrangler (Node.js, Cloudflare login)")
    parser.add_argument("--hf", metavar="OWNER/NAME", help="upload to this PRIVATE Hugging Face dataset "
                                                           "(a write token: HF_TOKEN or `hf auth login`)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    manifest = build(args.out)
    size = sum(f["bytes"] for f in manifest["files"].values())
    print(f"Built {args.out}: {len(manifest['files'])} files, {size / 1e6:.1f} MB, checked (no sealed runs, accounts "
          f"or audio). snapshot {manifest['snapshot_id']}, ratings up to {manifest['ratings_up_to']}, "
          f"data revision {manifest['data_revision']}.")
    if args.zip:
        made = shutil.make_archive(str(Path(args.out)), "zip", args.out)
        print(f"Zip for the dashboard upload: {made}")
    if args.hf:
        commit = hf_upload(args.out, args.hf, manifest)
        print(f"Uploaded to the private dataset {args.hf}: {commit.commit_url}\n"
              f"Readers: DRAMATIZER_RESULTS_URL=hf://datasets/{args.hf}  (pin this copy: @{commit.oid[:12]})")
    if args.deploy:
        return deploy(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
