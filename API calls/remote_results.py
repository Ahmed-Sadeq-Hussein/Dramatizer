"""remote_results.py | The same questions as API_REQUEST.py, answered from the hosted copy of the results, so
nobody's PC has to run the API.

The host publishes the results (no audio) with `python main_backend.py publish-results` to a team-only site
behind Cloudflare Access ("Access remote server.md"). This file downloads that copy once and answers the
evidence questions from it with the same code the API uses (backend/evidence_views.py), so an answer here is
the same as the API's for the same data. Every answer names its data: snapshot_id, ratings_up_to and
data_revision.

    from remote_results import connect
    results = connect()                                   # the address and token from the environment
    results.evidence_status()                             # as Results.evidence_status() against the API
    results.query_scores(group_by="gene_type", sort="mean")
    results.run_tool("answer_evidence", {"question": "Which genomes beat the plain line?"})

From the command line, as API_REQUEST.py:
    python remote_results.py evidence_status
    python remote_results.py query_scores '{"group_by": "listener"}'
    python remote_results.py manifest                     # which data the hosted copy holds

Where and who (environment variables):
    DRAMATIZER_RESULTS_URL        https://dramatizer-results.pages.dev, or a local folder of the copy
    DRAMATIZER_ACCESS_ID          a Cloudflare Access service token's Client ID      (for the hosted site)
    DRAMATIZER_ACCESS_SECRET      and its Client Secret
or, for a private Hugging Face dataset:
    DRAMATIZER_RESULTS_URL        hf://datasets/OWNER/NAME   (add @REVISION to read one published copy)
    DRAMATIZER_HF_TOKEN           your own Hugging Face read token (HF_TOKEN, or a login saved by
                                  `hf auth login`, work too)

Answered from the copy: evidence_status, query_scores, evidence_question, search_evidence, answer_evidence,
baseline_results, milestone2_results and about_dramatizer. The rest (searches, piles, renders, voices) need
the live API, and say so. The copy holds every listener's scores, as an admin sees them; it is as new as the
last publish. Needs the project's code (this repository) and numpy; nothing else.
"""
import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlsplit
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from API_REQUEST import ApiError, Results  # noqa: E402

from backend import evidence_views as views  # noqa: E402

DEFAULT_URL = "https://dramatizer-results.pages.dev"
LIVE_ONLY = "This is answered by the live API only; the hosted copy holds the evidence and the measured results."


def _saved_hf_login():
    """The token `hf auth login` saved on this PC, if huggingface_hub is installed; else None."""
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    return get_token()


class RemoteSite:
    """Reads the published copy (a URL behind Cloudflare Access, or a folder) and answers GET routes like the API."""

    def __init__(self, where=None, access_id=None, access_secret=None, hf_token=None):
        self.where = (where or os.environ.get("DRAMATIZER_RESULTS_URL") or DEFAULT_URL).rstrip("/")
        self.headers = {"User-Agent": "dramatizer-remote-results"}
        self.hf = self.where.startswith("hf://")
        if self.hf:                                   # hf://datasets/OWNER/NAME[@revision]: a private dataset
            repo, _, revision = self.where[len("hf://"):].removeprefix("datasets/").partition("@")
            if repo.count("/") != 1:
                raise ApiError(f"{self.where}: write a Hugging Face dataset as hf://datasets/OWNER/NAME.")
            self.where = f"https://huggingface.co/datasets/{repo}/resolve/{revision or 'main'}"
            token = hf_token or os.environ.get("DRAMATIZER_HF_TOKEN") or os.environ.get("HF_TOKEN") or _saved_hf_login()
            if token:
                self.headers["Authorization"] = f"Bearer {token}"
        access_id = access_id or os.environ.get("DRAMATIZER_ACCESS_ID")
        access_secret = access_secret or os.environ.get("DRAMATIZER_ACCESS_SECRET")
        if access_id and access_secret:
            self.headers.update({"CF-Access-Client-Id": access_id, "CF-Access-Client-Secret": access_secret})
        self._files = {}
        self._loaded = None

    @property
    def is_url(self):
        return self.where.startswith(("http://", "https://"))

    # ------------------------------------------------------------ reading files --
    def read(self, name):
        """One file of the copy as bytes, checked against the manifest's sha256 (once the manifest is read)."""
        if name in self._files:
            return self._files[name]
        if self.is_url:
            request = Request(f"{self.where}/{name}", headers=self.headers)
            try:
                with urlopen(request, timeout=60) as response:
                    if "cloudflareaccess.com" in response.geturl():
                        raise ApiError("The site asked for a login: set DRAMATIZER_ACCESS_ID and DRAMATIZER_ACCESS_SECRET "
                                       "(a service token from the host), see 'Access remote server.md'.")
                    data = response.read()
            except HTTPError as error:
                if self.hf and error.code in (401, 403, 404):
                    raise ApiError(f"Hugging Face would not give {name} (HTTP {error.code}): the dataset name is wrong, "
                                   "or DRAMATIZER_HF_TOKEN is missing or has no access to it. A private dataset "
                                   "answers 'not found' to anyone without access.") from None
                if error.code in (401, 403):
                    raise ApiError("The site refused the service token (401/403): ask the host whether it is still "
                                   "valid and allowed in the Access policy.") from None
                raise ApiError(f"Could not read {name} from {self.where}: HTTP {error.code}.") from None
            except URLError as error:
                raise ApiError(f"Could not reach {self.where}: {error.reason}.") from None
        else:
            path = Path(self.where) / name
            if not path.is_file():
                raise ApiError(f"No {name} in {self.where}: is it a folder made by publish-results?")
            data = path.read_bytes()
        if name != "manifest.json":
            expected = (self.manifest()["files"].get(name) or {}).get("sha256")
            if expected and hashlib.sha256(data).hexdigest() != expected:
                raise ApiError(f"{name} does not match the manifest: the copy is half-uploaded or changed. Try again.")
        self._files[name] = data
        return data

    def read_json(self, name):
        text = self.read(name).decode("utf-8")
        if text.lstrip().startswith("<"):
            raise ApiError("The site answered with a web page instead of data: the login did not go through. "
                           "Check DRAMATIZER_ACCESS_ID and DRAMATIZER_ACCESS_SECRET.")
        return json.loads(text)

    def manifest(self):
        if "manifest" not in self._files:
            self._files["manifest"] = self.read_json("manifest.json")
        return self._files["manifest"]

    def evidence(self):
        """(records, BM25 index, genomes, snapshot), read once."""
        if self._loaded is None:
            from backend.rag import baseline

            records = [json.loads(line) for line in self.read("evidence_records.jsonl").decode("utf-8").splitlines()
                       if line.strip()]
            self._loaded = (records, baseline.BM25(records), self.read_json("genomes.json"),
                            self.read_json("snapshot.json"))
        return self._loaded

    # --------------------------------------------------- answering like the API --
    def json(self, method, path):
        """GET <route>?<query> as the API answers it (the routes API_REQUEST.Results asks)."""
        if method != "GET":
            raise ApiError("The hosted copy is read-only.")
        parts = urlsplit(path)
        route, args = parts.path.strip("/"), dict(parse_qsl(parts.query))
        try:
            return self._answer(route, args)
        except views.Refused as refused:
            raise ApiError(refused.message) from None

    def _answer(self, route, args):
        if route == "results":
            return self.read_json("results.json")
        if route == "about":
            from backend.api.about import ABOUT
            return ABOUT
        if route == "manifest":
            return {k: v for k, v in self.manifest().items() if k != "files"}
        if not route.startswith("evidence/"):
            raise ApiError(f"{route}: {LIVE_ONLY}")
        records, index, genomes, snapshot = self.evidence()
        provenance = views.provenance(records, snapshot)
        what = route.split("/")[1]
        if what == "snapshot":
            from backend.genome import evidence_snapshot as E
            out = E.summary(snapshot, args.get("voice") or None)
            if args.get("voice") and not out["voices"]:
                raise views.Refused(f"No rated voice {args['voice']!r}. Voices are written DATASET/SPEAKER, e.g. VCTK/p250.")
            return out
        if what == "scores":
            return {**views.scores(views.score_rows(records, genomes), args), "note": None}
        if what == "search":
            return {**views.search(records, index, args.get("q"), args.get("kind"), args.get("k")), "provenance": provenance}
        if what == "answer":
            return {**views.answer(records, index, args.get("q")), "provenance": provenance}
        if what == "questions":
            if args.get("from") or args.get("to"):
                raise views.Refused("The hosted copy has the charts for every score only; a date range needs the live API.")
            number = route.split("/")[2] if len(route.split("/")) > 2 else ""
            if not number.isdigit():
                raise views.Refused("Ask a question by number: evidence/questions/1 to 6.")
            charts = self.read_json("charts/charts.json")
            chart = next((c for c in charts["charts"] if str(c["number"]) == number), None)
            url = f"{self.where}/charts/{Path(chart['file']).name}" if chart and "file" in chart else None
            return views.question(number, charts, snapshot, chart_url=url)
        raise ApiError(f"{route}: {LIVE_ONLY}")

    def log_out(self, _user=None):
        """Nothing to log out of: a service token stays valid until the host revokes it."""


def connect(where=None, access_id=None, access_secret=None, hf_token=None):
    """Results (API_REQUEST.py) that read the hosted copy instead of a running API."""
    results = Results(RemoteSite(where, access_id, access_secret, hf_token))
    results._user = None
    return results


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    results = connect()
    if argv[0] == "manifest":
        try:
            print(json.dumps(results.api.json("GET", "manifest"), indent=2, ensure_ascii=False))
        except ApiError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        return 0
    answer = results.run_tool(argv[0], argv[1] if len(argv) > 1 else None)
    print(json.dumps(json.loads(answer), indent=2, ensure_ascii=False))
    return 1 if '"error"' in answer[:12] else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main(sys.argv[1:]))
