# CI blocks a broken change (Milestone 4, step 1)

**Pull request:** https://github.com/Ahmed-Sadeq-Hussein/Dramatizer/pull/7, branch `ci-evidence/broken-test`,
opened 24 Nov 2026 and closed without merging.

One assertion in `deploy/test_blue_green.py` was broken on purpose: the first release is expected in the green slot
instead of blue.

**CI run #6:** https://github.com/Ahmed-Sadeq-Hussein/Dramatizer/actions/runs/45430382079

| Job / step | Result |
|---|---|
| tests: install (CPU only, no models) | success |
| tests: lint (ruff) | success |
| tests: tests | **failure** (the broken assertion) |
| image | **skipped**: it needs `tests`, so no image was built |

The same change on `main` would have built nothing either: the `image` job only runs after `tests` passes.

Earlier red runs, while CI was being set up (23 Nov): a lint error (CI run #1), a failing test (CI run #2) and a
vulnerable package (Audit run #1). Green runs on `main`: CI #3 (e177638), #4 (c9be43f) and
#5 (fb5303a, https://github.com/Ahmed-Sadeq-Hussein/Dramatizer/actions/runs/45430018347).
