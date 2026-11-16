"""second_listen.py | A second listener for the genomes that matter: the top safe genomes of every voice.

    python main_backend.py second-listen                 # build the piles (no renders: the audio is copied)
    python main_backend.py second-listen --dry-run       # show what would be built
    python main_backend.py second-listen --per-voice 3 --batch second-listener-v1

Why: in mos-diverse-v1 every genome was heard by one listener, so the evidence snapshot can name no winner
(it needs at least MIN_LISTENERS on the top genome and the runner-up). Rating every genome again would cost
240 clips; the winner rule only looks at the top of each voice, so this pile holds, per voice, its plain
line and its PER_VOICE highest-scored *safe* genomes (all artifact checks passed, still renderable), as
the snapshot ranks them.

How:
- One new run per voice, in OUTPUT_DIR/genome_search/<batch>_p<N>_<voice>, with the same audio (copied),
  the same genome numbers and a run.json that says "second_listen_of": the original run. The snapshot
  (evidence_snapshot.build) adds these scores to the original genomes, so n becomes 2.
- Pile N holds the voices of the original pile N, and goes to a listener who did NOT rate them
  (ROTATE: the listener of pile N+1). Each listener gets about 8 voices x 4 clips = 32 clips.
- "blind": true in run.json: the Rate page shows no one else's scores for these runs, even to an admin.
- The plain line is in the pile, so "above plain" compares each listener with their own plain-line score.

Nothing is assigned here: the host gives each pile to its listener with the printed commands.
"""
import argparse
import shutil
import sys
from datetime import datetime

from backend.genome import evidence_snapshot as E
from backend.genome import ratings as R
from backend.genome import search as S

BATCH = "second-listener-v1"
SOURCE_BATCH = "mos-diverse-v1"
PER_VOICE = 3


def _listeners_of(runs):
    """{pile: sorted listener names who rated that pile of the source batch}."""
    out = {}
    for folder, run in runs:
        if run.get("batch") == SOURCE_BATCH and run.get("pile"):
            names = {w for w in R.load(folder)["scores"] if not R.is_test_rater(w)}
            out.setdefault(run["pile"], set()).update(names)
    return {p: sorted(n) for p, n in out.items()}


def plan(snapshot, runs, per_voice=PER_VOICE, batch=BATCH):
    """[{"voice", "source", "pile", "genomes", "new_run", "for_listener"}] for every voice with safe genomes."""
    by_id = {r["id"]: (f, r) for f, r in runs}
    heard = _listeners_of(runs)
    piles = sorted(heard)
    rows = []
    for voice in snapshot["voices"]:
        safe = sorted((g for g in snapshot["genomes"] if g["voice"] == voice["voice"] and g["rated"] and g["safe"]),
                      key=lambda g: (-g["mos"], g["genome"]))[:per_voice]
        if not safe:
            continue
        source = safe[0]["run"]
        _, run = by_id[source]
        pile = run.get("pile")
        if pile not in heard:
            continue
        others = [w for p in piles[piles.index(pile) + 1:] + piles[:piles.index(pile)] for w in heard[p]
                  if w not in heard[pile]]
        rows.append({"voice": voice["voice"], "source": source, "pile": pile,
                     "genomes": sorted(g["genome"] for g in safe if g["run"] == source),
                     "new_run": f"{batch}_p{pile}_{voice['voice'].replace('/', '_')}",
                     "for_listener": others[0] if others else None})
    return rows


def build(rows, runs, batch=BATCH, runs_dir=None):
    """Make each planned run folder (skipped when it exists). Returns the folders made."""
    by_id = {r["id"]: (f, r) for f, r in runs}
    root = (runs_dir or S.RUNS_DIR)
    made = []
    for row in rows:
        folder = root / row["new_run"]
        if (folder / "run.json").is_file():
            continue
        src_folder, src = by_id[row["source"]]
        folder.mkdir(parents=True, exist_ok=True)
        genomes = [g for g in src["genomes"] if g["number"] in row["genomes"]]
        shutil.copy2(src_folder / "baseline.wav", folder / "baseline.wav")
        for g in genomes:
            name = g.get("audio") or f"{R.clip_key(g['number'])}.wav"
            shutil.copy2(src_folder / name, folder / name)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run = {k: v for k, v in src.items() if k not in ("genomes", "requested_by")}
        run.update(id=row["new_run"], batch=batch, pile=row["pile"], genomes=genomes, created=now, finished=now,
                   status="finished", method="second-listen", method_label="Second listener for the top genomes",
                   second_listen_of=row["source"], blind=True, for_listener=row["for_listener"])
        S.save_run(folder, run)
        made.append(folder)
    return made


def main(argv=None):
    parser = argparse.ArgumentParser(description="Piles that give the top safe genomes of every voice a second listener.")
    parser.add_argument("--per-voice", type=int, default=PER_VOICE, help=f"genomes per voice (default {PER_VOICE})")
    parser.add_argument("--batch", default=BATCH, help=f"the new batch's name (default {BATCH})")
    parser.add_argument("--dry-run", action="store_true", help="only show the plan")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    runs = S.list_runs()
    source = [(f, r) for f, r in runs if r.get("batch") == SOURCE_BATCH]
    snapshot = E.build(source)
    rows = plan(snapshot, source, args.per_voice, args.batch)
    print(f"Snapshot {snapshot['provenance']['snapshot_id']}: {len(rows)} voices, "
          f"{sum(len(r['genomes']) for r in rows)} genomes + {len(rows)} plain lines.")
    for pile in sorted({r["pile"] for r in rows}):
        mine = [r for r in rows if r["pile"] == pile]
        who = sorted({r["for_listener"] for r in mine if r["for_listener"]})
        print(f"\nPile {pile} for {', '.join(who) or 'anyone who did not rate it'}: "
              f"{len(mine)} voices, {sum(len(r['genomes']) + 1 for r in mine)} clips")
        for r in mine:
            print(f"  {r['voice']}: genomes {', '.join(map(str, r['genomes']))} of {r['source']}")
    if args.dry_run:
        print("\nDry run: nothing written.")
        return
    made = build(rows, runs, args.batch)
    print(f"\n{len(made)} run folders made in {S.RUNS_DIR} ({len(rows) - len(made)} already there).")
    from backend.api import accounts

    login = {a.display_name: a.username for a in accounts.load().values()}
    print("Give each pile to its listener (the host runs these):")
    for pile in sorted({r["pile"] for r in rows}):
        who = next((r["for_listener"] for r in rows if r["pile"] == pile and r["for_listener"]), None)
        if who:
            print(f"  python main_backend.py account assign {login.get(who, who)} {args.batch} {pile}")


if __name__ == "__main__":
    main()
