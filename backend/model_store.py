"""model_store.py | Every model the site loads, pinned by revision, in one folder: the model store (MODELS_DIR).

    python main_backend.py models fetch [--only NAME]   # fill the store (D:/Dramatizer models by default)
    python main_backend.py models check [--full]        # every file there, the right size (--full: and sha256)
    python main_backend.py models pin NAME              # which revision a local copy matches (to set a pin)

The store's layout is what the container mounts at /models (deploy/container.env):

    Fun-CosyVoice3-0.5B/                     COSYVOICE_MODEL_DIR
    llm/gemma-4-E4B-it/, llm/Qwen3-4B/, ...  LLM_LIBRARY_DIR
    cache/whisper/base.pt                    XDG_CACHE_HOME=<store>/cache (Whisper looks there)
    cache/torch/hub/checkpoints/squim_*.pth  TORCH_HOME=<store>/cache/torch (torchaudio looks there)
    cache/wetext/{en,zh}/tn/*.fst            WETEXT_DIR (CosyVoice's text normaliser, never fetched at load)
    models.json                              each model's source, revision, and every file's size and sha256

fetch never trusts a file it has not checked: a Hugging Face file must match the pinned revision (its sha256 for
large files, its git blob hash for small ones), and Whisper and SQUIM files must match the sha256 pinned here. A
file already on this computer (the folders in .env) that matches is hard-linked into the store, so it takes no
extra space; anything else is downloaded at the pinned revision. Nothing is ever downloaded at start-up: the server
checks the store (check(), sizes only, a few seconds) and refuses to start when a model is missing.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from backend import model

DEFAULT_STORE = "D:/Dramatizer models"
STORE_FILE = "models.json"

# The pins. Each Hugging Face revision is a full commit hash; `models pin NAME` finds the one a local copy matches.
MODELS = {
    "Fun-CosyVoice3-0.5B": {
        "repo": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "revision": "29e01c4e8d000f4bcd70751be16fa94bf3d85a18",
        "folder": "Fun-CosyVoice3-0.5B", "local": ("COSYVOICE_MODEL_DIR", None),
        "why": "the voice model (every render)",
    },
    "gemma-4-E4B-it": {
        "repo": "google/gemma-4-E4B-it",
        "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",    # 20 Jul 2026: before the 3 Nov tokenizer change
        "folder": "llm/gemma-4-E4B-it", "local": ("LLM_LIBRARY_DIR", "gemma-4-E4B-it"),
        "why": "the chatbot (its own Python environment)",
    },
    "Qwen3-4B": {
        "repo": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "folder": "llm/Qwen3-4B", "local": ("LLM_LIBRARY_DIR", "Qwen3-4B"),
        "why": "emphasis scoring (EMPHASIS_MODEL)",
    },
    "all-MiniLM-L6-v2": {
        "repo": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "folder": "llm/all-MiniLM-L6-v2", "local": ("LLM_LIBRARY_DIR", "all-MiniLM-L6-v2"),
        "files": ["config.json", "model.safetensors", "special_tokens_map.json", "tokenizer.json",
                  "tokenizer_config.json", "vocab.txt"],       # what rag/dense.py loads with transformers
        "why": "dense retrieval for the chatbot (the RAG comparison)",
    },
    "whisper-base": {
        "url": "https://openaipublic.azureedge.net/main/whisper/models/"
               "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
        "sha256": "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e",
        "folder": "cache/whisper", "file": "base.pt", "local": ("HOME_CACHE", "whisper"),
        "why": "word error rate, and the check of uploaded voices",
    },
    "squim-objective": {
        "url": "https://download.pytorch.org/torchaudio/models/squim_objective_dns2020.pth",
        "sha256": "2c54586fea83fb5eb5394d710038ee89f55cab7011a5bf730bebed4c8777e828",   # torchaudio publishes
        # no checksum: this is the copy torchaudio downloaded here (`models pin squim-objective`)
        "folder": "cache/torch/hub/checkpoints", "file": "squim_objective_dns2020.pth",
        "local": ("TORCH_CACHE", "hub/checkpoints"),
        "why": "the quality estimate (SQUIM)",
    },
    "wetext": {
        # CosyVoice's text normaliser ("12" -> "twelve"). wetext fetches its whole repository (236 MB, every
        # language) from ModelScope at each load, which now refuses anonymous requests; without the files CosyVoice
        # goes on with NO normaliser. Not on Hugging Face, so each file is pinned by sha256 (the ModelScope commit
        # it came from is kept too). Only English and Chinese text normalisation is loaded (frontend.py).
        "modelscope": "pengzhendong/wetext",
        "files": {
            "en/tn/tagger.fst": {"sha256": "ee9353d503e20d6dd89508588ad2e261c837046ca38fac878faeb1e433abcb08",
                                 "commit": "887eec2274a4502b9252185d0e6ee4614687c475"},
            "en/tn/verbalizer.fst": {"sha256": "787142a2085a8eda20cfea2cd6295441347aadd6ed13de3a9905f2bd8b64ac8c",
                                     "commit": "887eec2274a4502b9252185d0e6ee4614687c475"},
            "zh/tn/tagger.fst": {"sha256": "cf341314c51f7ce59049f3b2c42f0ce8fd71e6d08d4d6969613aad384a5e2ae8",
                                 "commit": "b04bc07588601f7619b20efbb01cd1fa7278ccbc"},
            "zh/tn/verbalizer.fst": {"sha256": "5a13cd679dd54637d12d2bd1bd33ee2165d91c867e14468c93195af02256e5da",
                                     "commit": "b04bc07588601f7619b20efbb01cd1fa7278ccbc"},
        },
        "folder": "cache/wetext", "local": ("MODELSCOPE_CACHE", "pengzhendong/wetext"),
        "why": "CosyVoice's text normaliser (WETEXT_DIR)",
    },
}


class StoreError(Exception):
    """The store is missing a model, or a file does not match its pin."""


def store_dir():
    return Path(model.setting("MODELS_DIR", DEFAULT_STORE))


def pinned_revision(spec):
    """The one string that names what a model is pinned to: a Hugging Face commit, a file's sha256, or, for a
    model pinned file by file (ModelScope), the sha256 of its sorted "path sha256" lines."""
    if spec.get("revision") or spec.get("sha256"):
        return spec.get("revision") or spec.get("sha256")
    if spec.get("modelscope"):
        lines = "".join(f"{relative} {want['sha256']}\n" for relative, want in sorted(spec["files"].items()))
        return hashlib.sha256(lines.encode()).hexdigest()
    return None


# --------------------------------------------------------------------------- #
# Hashes
# --------------------------------------------------------------------------- #


def sha256_of(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while block := file.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def git_blob_of(path):
    """The git blob hash, which Hugging Face gives for files that are not in LFS (the small ones)."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def matches(path, expected):
    """Does the file at path match expected = {"size", "sha256"?, "blob"?}? Size first, so a wrong file is cheap."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size != expected["size"]:
        return False
    if expected.get("sha256"):
        return sha256_of(path) == expected["sha256"]
    return git_blob_of(path) == expected["blob"]


# --------------------------------------------------------------------------- #
# What each model should hold
# --------------------------------------------------------------------------- #


def remote_files(name, spec, api=None):
    """{relative path: {"size", "sha256" or "blob"}} of a Hugging Face model at its pinned revision."""
    from huggingface_hub import HfApi

    api = api or HfApi()
    info = api.model_info(spec["repo"], revision=spec["revision"], files_metadata=True)
    wanted = set(spec.get("files") or [])
    out = {}
    for sibling in info.siblings:
        if wanted and sibling.rfilename not in wanted:
            continue
        if not wanted and (sibling.rfilename.startswith(".") or sibling.rfilename.endswith(".md")):
            continue
        lfs = sibling.lfs or {}
        sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        out[sibling.rfilename] = {"size": sibling.size, "sha256": sha} if sha else \
            {"size": sibling.size, "blob": sibling.blob_id}
    missing = wanted - set(out)
    if missing:
        raise StoreError(f"{name}: {sorted(missing)} are not in {spec['repo']} at {spec['revision'][:10]}.")
    return out


def local_folder(spec):
    """Where this computer already keeps the model (from .env), or None."""
    setting, sub = spec["local"]
    if setting == "HOME_CACHE":
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    elif setting == "TORCH_CACHE":
        base = Path(os.environ.get("TORCH_HOME") or Path.home() / ".cache" / "torch")
    elif setting == "MODELSCOPE_CACHE":
        base = Path(os.environ.get("MODELSCOPE_CACHE") or Path.home() / ".cache" / "modelscope" / "hub")
    else:
        value = model.setting(setting, "")
        if not value:
            return None
        base = Path(value)
    folder = base / sub if sub else base
    return folder if folder.is_dir() else None


def place(source, target):
    """Put source at target: a hard link when they share a drive (no extra space), else a copy."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


# --------------------------------------------------------------------------- #
# fetch, check, pin
# --------------------------------------------------------------------------- #


def fetch_one(name, spec, store, api=None, download=None, say=print):
    """Fill one model's folder in the store. Returns its models.json entry."""
    target = Path(store) / spec["folder"]
    local = local_folder(spec)
    if "repo" in spec:
        expected = remote_files(name, spec, api)
    elif "modelscope" in spec:
        expected = {}
        for relative, want in spec["files"].items():
            source = local / relative if local else None
            expected[relative] = {"size": source.stat().st_size if source and source.is_file() else None,
                                  "sha256": want["sha256"]}
    else:
        sha = spec["sha256"]
        if not sha:
            raise StoreError(f"{name} has no sha256 pin yet: run `models pin {name}`.")
        source = local / spec["file"] if local else None
        size = source.stat().st_size if source and source.is_file() else None
        expected = {spec["file"]: {"size": size, "sha256": sha}}

    files = {}
    for relative, want in sorted(expected.items()):
        here = target / relative
        how = "kept"
        if not (want["size"] is not None and matches(here, want)):
            there = local / relative if local else None
            if there and want["size"] is not None and matches(there, want):
                place(there, here)
                how = "linked"
            else:
                (download or _download)(name, spec, relative, here)
                how = "downloaded"
                want = {**want, "size": here.stat().st_size}
                if not matches(here, want):
                    here.unlink()
                    raise StoreError(f"{name}/{relative}: the download does not match the pin.")
        files[relative] = {"size": here.stat().st_size,
                           "sha256": want.get("sha256") or sha256_of(here)}
        say(f"  {name}/{relative}: {how}")
    entry = {"folder": spec["folder"], "why": spec["why"], "files": files}
    if "repo" in spec:
        entry.update(source="huggingface", repo=spec["repo"], revision=spec["revision"])
    elif "modelscope" in spec:
        entry.update(source="modelscope", repo=spec["modelscope"], revision=pinned_revision(spec),
                     commits={relative: want["commit"] for relative, want in spec["files"].items()})
    else:
        entry.update(source="url", repo=spec["url"], revision=spec["sha256"])
    return entry


def _download(name, spec, relative, here):
    here.parent.mkdir(parents=True, exist_ok=True)
    if "repo" in spec:
        from huggingface_hub import hf_hub_download

        root = here
        for _ in Path(relative).parts:
            root = root.parent              # the model's folder: here minus the relative path
        made = hf_hub_download(spec["repo"], relative, revision=spec["revision"], local_dir=str(root))
        if Path(made).resolve() != here.resolve():
            shutil.move(made, here)
    elif "modelscope" in spec:
        import urllib.request

        commit = spec["files"][relative]["commit"]
        urllib.request.urlretrieve(f"https://www.modelscope.cn/api/v1/models/{spec['modelscope']}/repo"
                                   f"?Revision={commit}&FilePath={relative}", here)
    else:
        import urllib.request

        urllib.request.urlretrieve(spec["url"], here)


def write_store(store, entries, now=None):
    path = Path(store) / STORE_FILE
    old = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"models": {}}
    old["models"].update(entries)
    data = {"made": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "what": "Every model the site loads, pinned. Written by `python main_backend.py models fetch`.",
            "models": dict(sorted(old["models"].items()))}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


def fetch(store=None, only=None, api=None, download=None, say=print):
    store = Path(store or store_dir())
    store.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name, spec in MODELS.items():
        if only and name not in only:
            continue
        say(f"{name} ({spec['why']}):")
        entries[name] = fetch_one(name, spec, store, api, download, say)
    return write_store(store, entries)


def check(store=None, full=False, names=None):
    """Every model in models.json is there with every file the right size (full: and the right sha256), and every
    model in MODELS (or names) is in models.json at its pinned revision. Returns a list of problems."""
    store = Path(store or store_dir())
    path = store / STORE_FILE
    if not path.is_file():
        return [f"No model store in {store} ({STORE_FILE} is missing): run `python main_backend.py models fetch`."]
    recorded = json.loads(path.read_text(encoding="utf-8")).get("models", {})
    problems = []
    for name in names or MODELS:
        spec, entry = MODELS.get(name, {}), recorded.get(name)
        if entry is None:
            problems.append(f"{name} is not in the store.")
            continue
        pinned = pinned_revision(spec)
        if pinned and entry.get("revision") != pinned:
            problems.append(f"{name} is at {str(entry.get('revision'))[:10]}, the pin is {pinned[:10]}.")
        for relative, want in entry["files"].items():
            here = store / entry["folder"] / relative
            if not here.is_file():
                problems.append(f"{name}/{relative} is missing.")
            elif here.stat().st_size != want["size"]:
                problems.append(f"{name}/{relative} has the wrong size.")
            elif full and sha256_of(here) != want["sha256"]:
                problems.append(f"{name}/{relative} has changed (sha256 differs).")
    return problems


def pin(name, api=None):
    """The newest revision of a Hugging Face model that this computer's copy matches file for file (or, for a
    file pinned by sha256, that file's sha256). Prints it; set it in MODELS by hand."""
    from huggingface_hub import HfApi

    spec = MODELS[name]
    local = local_folder(spec)
    if local is None:
        raise StoreError(f"{name}: no local copy to pin from.")
    if "modelscope" in spec:
        return {relative: sha256_of(local / relative) for relative in spec["files"]}
    if "repo" not in spec:
        return sha256_of(local / spec["file"])
    api = api or HfApi()
    for commit in api.list_repo_commits(spec["repo"]):
        try:
            files = remote_files(name, {**spec, "revision": commit.commit_id}, api)
        except StoreError:
            continue
        if all(matches(local / relative, want) for relative, want in files.items()):
            return commit.commit_id
    raise StoreError(f"{name}: the copy in {local} matches no revision of {spec['repo']}.")


def settings_for(store):
    """The .env lines that make this computer load every model from the store."""
    store = Path(store).as_posix()
    return [f"MODELS_DIR={store}", f"COSYVOICE_MODEL_DIR={store}/Fun-CosyVoice3-0.5B", f"LLM_LIBRARY_DIR={store}/llm",
            f"XDG_CACHE_HOME={store}/cache", f"TORCH_HOME={store}/cache/torch", f"WETEXT_DIR={store}/cache/wetext"]


def check_at_start():
    """Called by the server at start-up when MODELS_DIR is set: stop with the problems instead of serving without
    a model (or downloading one)."""
    if not model.setting("MODELS_DIR", ""):
        return
    needed = [name for name in MODELS if name != "gemma-4-E4B-it" or model.setting("CHATBOT_PYTHON", "")]
    problems = check(names=needed)
    if problems:
        raise SystemExit("The model store is not complete, so the server does not start:\n  " +
                         "\n  ".join(problems[:10]))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="models", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    one = sub.add_parser("fetch", help="fill the store, by pinned revision")
    one.add_argument("--only", nargs="*", choices=list(MODELS), help="just these models")
    one.add_argument("--to", help=f"the store (default MODELS_DIR, else {DEFAULT_STORE})")
    one = sub.add_parser("check", help="every model there, every file the right size")
    one.add_argument("--full", action="store_true", help="also every file's sha256 (minutes)")
    one.add_argument("--to", help="the store")
    one = sub.add_parser("pin", help="the revision this computer's copy of a model matches")
    one.add_argument("name", choices=list(MODELS))
    args = parser.parse_args(argv)

    try:
        if args.action == "fetch":
            store = Path(args.to or store_dir())
            fetch(store, args.only)
            problems = check(store, names=args.only)
            if problems:
                raise StoreError("; ".join(problems))
            print(f"\nThe store in {store} is complete. To load every model from it, put in .env:\n  " +
                  "\n  ".join(settings_for(store)))
        elif args.action == "check":
            store = Path(args.to or store_dir())
            problems = check(store, full=args.full)
            for problem in problems:
                print("  " + problem)
            print(f"{store}: " + (f"{len(problems)} problems" if problems else "complete"))
            return 1 if problems else 0
        elif args.action == "pin":
            print(pin(args.name))
    except StoreError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
