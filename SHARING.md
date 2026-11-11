# Sharing Dramatizer with the group

The heavy work (the voice model, the GPU, the voice data) stays on **one PC: the
host's**. Everyone else runs a small **client** on their own PC and uses it in
their browser. The client talks to the host's **API** through a playit.gg tunnel.

```
 Their PC                                    The host's PC
 ┌───────────────────────────┐   internet    ┌──────────────────────────────────┐
 │ browser → client (5010)   │ ───────────▶  │ playit ─▶ API (5005) ─▶ models,  │
 │ python main_frontend.py   │  HTTPS, key,  │ python main_backend.py api       │
 │   connect https://...     │  pinned cert  │   GPU, voices, results           │
 └───────────────────────────┘               └──────────────────────────────────┘
```

## The host (once)

1. Give each person a key. It is shown once; send it to them privately:

   ```
   python main_backend.py api-key add Ana
   ```

2. Start the API. It prints the **fingerprint** of its certificate:

   ```
   python main_backend.py api
   ```

3. In playit.gg, add a **TCP** tunnel to local address `127.0.0.1`, port `5005`.
   Tunnel only this port.

4. Send each person three things: the playit address (`https://name.playit.gg:12345`),
   their key, and the fingerprint.

Other commands: `api-key list` shows who has a key, and `api-key remove Ana` stops
Ana's key at once. `api --fingerprint` prints the fingerprint again.

### Accounts: a username and password instead of a key

Listeners log in with an account. Its role decides what it may do: a
**listener** rates genomes (only as themself) and sees rankings; a
**researcher** may also speak lines, start searches and name genomes; an
**admin** may also save scores for other listeners.

```
python main_backend.py account add julius --name Julius --password-file history/api/passwords_to_hand_out.txt
python main_backend.py account list
python main_backend.py account reset julius      # a new password; their open sessions end
python main_backend.py account disable julius    # or: enable, remove, role julius researcher
python main_backend.py account assign julius mos-diverse-v1 2   # julius rates listening pile 2
```

With a pile assigned, a listener sees and rates only that pile. Admins see every
pile and start on their own. The first listening batch is split as: pile 1
Ash, pile 2 Julius, pile 3 Rama.

Listeners rate **blind**. Their page has only the **Rate genomes** tab. They
see their own pile and their own scores: no other searches, no other pile's
audio, no other listener's scores, and no MOS, because the MOS would show what
the others gave. Admins and researchers see everything.

Only a salted PBKDF2 hash of each password is kept (`history/api/accounts.json`).
After 5 wrong passwords an account is locked for 15 minutes. Send each person
their username and password privately, then delete the password file. They can
change their password under **Account** on the page.

## Everyone else

1. Get the project folder, and Python (3.10 or newer) with two packages. The
   client needs only `main_frontend.py` and the `frontend/` folder: no models,
   no GPU, no voice data.

   ```
   git clone https://github.com/Ahmed-Sadeq-Hussein/Dramatizer.git
   ```

   ```
   pip install flask requests
   ```

   If you already have the folder, `git pull` gets the newest page.

2. Connect:

   ```
   python main_frontend.py connect https://name.playit.gg:12345
   ```

   The first time, it shows the host's fingerprint. Type `yes` only if it is
   exactly the one the host sent you. Then paste your key, or, if the host gave
   you a username and password, press Enter and log in on the page. The address
   and fingerprint are remembered, so next time `python main_frontend.py connect`
   is enough. A password is never saved.

3. The page opens at http://127.0.0.1:5010. There are three steps:
   **1. Choose a voice**, **2. Speak** a line, and **3. Genomes**: search, rate
   with up to 4 listeners, rank by MOS, and name the best genome. **Results**
   shows the measured baselines.

   **Rate genomes** is where anyone with a pile starts after logging in, with
   their own pile already open. For each voice, score the plain line and every
   version from 0 to 5 in steps of 0.5. Each score is saved as soon as it is
   set, and the card shows when ("Saved 10 Nov, 14:03"). The host reads each
   score back from disk before the card says Saved. If a save fails (no
   connection, the tunnel is down), the card says "Not saved yet". The score is
   kept in the browser and sent again every few seconds, and again the next time
   the page opens. A rated clip's card is green and reads "✓ Rated 3.5 · 10 Nov,
   14:03"; one still to do is white and reads "○ Not rated yet". Each voice says
   how many are left, and **Next unrated** jumps to the next one, across the pile.
   **Check with host** asks the host what it has kept and shows that. If anything
   differs, the page corrects it and says how many. **Ranking** shows every version you scored, best first, with
   a bar per score, the date you scored it, and a mark for the plain line of the
   same voice. You can play any of them again and **Re-score** it.

### Who gave which score, and when

Everything is kept on the host, in `history/` (git-ignored, since ratings are
personal data):

| File | What is in it |
|---|---|
| `history/ratings/who_rated_what.csv` | The current score of every listener for every clip: listener, account, batch, pile, voice, run, clip, genome, score, score minus their plain-line score, date last scored, date first scored, number of changes. Rebuilt after every save, so it opens in a spreadsheet as it stands. |
| `history/ratings/mos.jsonl` | Every score ever given or changed, one line each, never rewritten. It includes the score it replaced. |
| `<run folder>/ratings.json` | That run's scores, with the same date and account details per score. |
| `history/api/audit.jsonl` | Logins, refused logins, logouts, password changes and saves: when, who, and from where. Never passwords or tokens. |

```bash
python main_backend.py ratings --who
```

```bash
python main_backend.py ratings --current scores.csv
```

```bash
python main_backend.py ratings --export every_change.csv
```

`--who` prints how many clips each listener has scored per pile, from their
first to their last score.

## The day of a listening session (host checklist)

1. Commit and push the client (`main_frontend.py`, `frontend/client/`, including
   `static/rate.js`), so the others' `git pull` gets the rating page.
2. Start the API and leave the window open:

   ```bash
   python main_backend.py api
   ```

3. Start the playit tunnel to `127.0.0.1:5005`, and check that its address is
   the one you sent.
4. Send each person, privately: the playit address, the fingerprint the API
   printed, their username and their password. Then delete
   `history/api/passwords_to_hand_out.txt`.
5. While they rate: `python main_backend.py ratings --who` shows how far each
   one is, and `history/ratings/who_rated_what.csv` has every score with its date.

If someone is locked out after wrong passwords, it lifts after 15 minutes, or
`python main_backend.py account reset <username>` gives them a new password.

## Why it is safe enough

- **Every call needs a key or a login.** Each person has their own, and the host can
  remove it. Passwords are stored only as salted hashes, and the session token stays
  in the client program, never in the browser.
- **HTTPS with a pinned certificate.** The client only talks to the host's own
  certificate, so nobody in between can read or change the traffic.
- **The API can only do a fixed list of things.** It cannot run commands or read
  files, and it picks voices by id.
- **Only validation voices are offered.** The final test speakers and personal
  custom voices never are.
- **One GPU job at a time, in a queue,** with at most 3 waiting per person.
- **Guessing is slow.** Each wrong login waits a second. An account locks for 15 minutes
  after 5 wrong passwords. Once there are more than 30 wrong logins on the server in
  10 minutes (someone trying many names), every login is held back until it is quiet again.
- **Strict browser rules.** The host's replies and the client's page cannot be put
  inside another site's frame, are never cached, and send no referrer. The page only
  loads its own files. The client answers only at 127.0.0.1 or localhost, and only
  its own page can save through it. Bodies are capped at 64 KB, and the host does not
  say what software it runs.
- **Every call is logged** with who made it, in `history/api/access.log`. Logins and
  saved scores are also recorded in `history/api/audit.jsonl`.

Limits: it works only while the host's PC, the API and the tunnel are running.
The cloned audio of real people stays within the group.

## If something goes wrong

| Message | What to do |
|---|---|
| `STOP: the host's certificate is not the one you trusted` | Ask the host whether they made a new certificate. If they did, run `connect --forget` and connect again. If they did not, do not connect. |
| `Wrong username or password` | Check both. After 5 wrong tries the account is locked for 15 minutes; the host can `account reset` it. |
| `The host does not accept this key` | Ask the host for a new key, then run `connect --forget` and connect again. |
| `The host cannot be reached` | The host's API or the playit tunnel is not running. |
| `You already have 3 jobs waiting` | Wait for one of them to finish. |
