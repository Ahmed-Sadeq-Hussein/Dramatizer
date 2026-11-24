"""release.py | Write deploy/release.json for a release candidate (Milestone 4, step 3): what it is meant to run.

    python deploy/release.py v8 --image ghcr.io/...@sha256:... --base ghcr.io/...@sha256:... \
        --env deploy/container.env --voice-pack "D:/Dramatizer voice pack/<revision>" --models "D:/Dramatizer models"

    python deploy/release.py host-2026-11-23 --env .env        # the site as started today, without Docker

The settings hash is made from the env file the candidate is started with (only the keys in
backend/release_info.CONFIG_KEYS; nothing secret is read out or written). The voice pack's and models' revisions
are read from the folders on THIS computer that the container will mount, because the paths inside the env file
are the container's (/voices, /models). deploy/smoke.py then compares this file with the candidate's /version.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import release_info  # noqa: E402

OUT = ROOT / "deploy" / "release.json"


def make(name, env_file, image=None, base=None, voice_pack_dir=None, models_dir=None, commit=None):
    values = release_info.read_env_file(env_file)
    return {
        "release": name,
        "commit": commit or release_info.commit(),
        "image": image,
        "base_image": base,
        "voice_pack": release_info.voice_pack(voice_pack_dir or values.get("VOICE_PACK_DIR")),
        "models": release_info.models(models_dir or values.get("MODELS_DIR")),
        "config_sha256": release_info.config_sha256(values),
        "made": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write deploy/release.json for a release candidate.")
    parser.add_argument("name", help="the release's name, e.g. v8")
    parser.add_argument("--env", default=str(ROOT / "deploy" / "container.env"),
                        help="the env file it is started with (default deploy/container.env)")
    parser.add_argument("--image", help="the app image, by digest (ghcr.io/...@sha256:...)")
    parser.add_argument("--base", help="the base image, by digest")
    parser.add_argument("--voice-pack", help="the voice pack folder on this computer (mounted at /voices)")
    parser.add_argument("--models", help="the model store on this computer (mounted at /models)")
    parser.add_argument("--commit", help="default: git's HEAD here")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)
    for digest in (args.image, args.base):
        # A registry digest (…@sha256:…), or before the image is pushed, its local ID (sha256:…): both name exactly
        # one image's content, which a tag does not
        if digest and "@sha256:" not in digest and not digest.startswith("sha256:"):
            raise SystemExit(f"Name images by digest (…@sha256:…) or local ID (sha256:…), never by a tag: {digest}")
    release = make(args.name, args.env, args.image, args.base, args.voice_pack, args.models, args.commit)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}: {release['release']} at commit {release['commit'][:12]}, "
          f"settings {release['config_sha256'][:12]}")


if __name__ == "__main__":
    main()
