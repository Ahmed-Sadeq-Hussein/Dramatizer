"""main_frontend.py | One place to start the front end: the website and logging in.

    python main_frontend.py connect https://ADDRESS:PORT   # on anyone's PC: use the host's API (frontend/client/)
    python main_frontend.py             # the current site, V4, on the host's PC: login, HTTPS, phone layout
    python main_frontend.py v3          # an older version, without a login (this PC only)
    python main_frontend.py --list      # every version

The pages live in frontend/:

    frontend/ui_versions/     every version of the site; each builds on the one before
    frontend/templates/       the HTML of the first two versions
    frontend/static/          their CSS and JavaScript

Only V4 has a login (UI_PASSWORD in .env; one is made on the first start if it is
missing). The older versions have none, so they only listen on this PC.

The site calls the back end (backend/) for everything else: the voices, the
models and the measurements. The back end's own commands are run by main_backend.py.
"""
import runpy
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

VERSIONS = {
    "connect": ("On anyone's PC: use the host's API in the browser (connect https://ADDRESS:PORT)",
                "frontend.client.client"),
    "v4":("The current site: password login, HTTPS, phone layout, random voices, Examine",
           "frontend.ui_versions.Voice_model_example_UI_V4"),
    "v3": ("V2, plus cleaning and combining the reference", "frontend.ui_versions.Voice_model_example_UI_V3"),
    "v2": ("Picking voices with Picky_voice's options", "frontend.ui_versions.Voice_model_example_UI_V2"),
    "v1": ("Pick a voice, then make it say something", "frontend.ui_versions.Voice_model_example_UI"),
    "library": ("Browse the voice library only, no speech", "frontend.ui_versions.Voice_Importer_UI"),
}
CURRENT = "v4"


def main(argv):
    if argv and argv[0] in ("-h", "--help", "--list", "help"):
        print(__doc__.split("\n\nThe pages live")[0].split("\n", 1)[1])
        print("\nVersions:")
        for name, (what, _) in VERSIONS.items():
            print(f"  {name:8s} {what}{'  (default)' if name == CURRENT else ''}")
        return 0

    name = argv[0].lower() if argv else CURRENT
    if name not in VERSIONS:
        print(f"Unknown version {name!r}: use one of {', '.join(VERSIONS)}.", file=sys.stderr)
        return 2

    sys.argv = [VERSIONS[name][1], *argv[1:]]
    runpy.run_module(VERSIONS[name][1], run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_DIR))
    sys.exit(main(sys.argv[1:]))
