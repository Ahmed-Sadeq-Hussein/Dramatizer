"""make_manifest.py | List every clip in the datasets once, in a fixed order.

    Writes history/eval/manifest.csv. The readers come from Voice_Importer.py, so
    run this in the dramatizer environment (it needs pyarrow).
"""
import csv
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
load_dotenv(PROJECT / ".env")

import Voice_Importer as library  # noqa: E402

EVAL_DIR = PROJECT / "history" / "eval"
MANIFEST = EVAL_DIR / "manifest.csv"
FIELDS = ["dataset", "clip_id", "speaker", "text", "emotion", "intensity", "mic", "licence", "source"]
LICENCE = {"CREMA-D": "ODbL", "VCTK": "CC BY 4.0"}


def main():
    library_dir = Path(os.environ["VOICE_LIBRARY_DIR"])
    clips = (
        library.read_crema_d(library_dir / os.environ.get("CREMA_D_FOLDER", "CREMA-D"))
        + library.read_vctk(library_dir / os.environ.get("VCTK_FOLDER", "VCTK"))
    )
    clips.sort(key=lambda c: (c.dataset, c.speaker, c.clip_id))

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for c in clips:
            writer.writerow({
                "dataset": c.dataset,
                "clip_id": c.clip_id,
                "speaker": c.speaker,
                "text": c.text,
                "emotion": c.details.get("emotion", ""),
                "intensity": c.details.get("intensity", ""),
                "mic": c.details.get("mic", ""),
                "licence": LICENCE[c.dataset],
                "source": str(c.source),
            })
    print(f"{len(clips)} clips -> {MANIFEST}")


if __name__ == "__main__":
    main()
