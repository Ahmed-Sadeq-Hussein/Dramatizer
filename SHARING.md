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

## Everyone else

1. Get the project folder, and Python with two packages:

   ```
   pip install flask requests
   ```

2. Connect:

   ```
   python main_frontend.py connect https://name.playit.gg:12345
   ```

   The first time, it shows the host's fingerprint. Type `yes` only if it is
   exactly the one the host sent you. Then paste your key. Both are remembered,
   so next time `python main_frontend.py connect` is enough.

3. The page opens at http://127.0.0.1:5010. There are three steps:
   **1. Choose a voice**, **2. Speak** a line, and **3. Genomes**: search, rate
   with up to 4 listeners, rank by MOS, and name the best genome. **Results**
   shows the measured baselines.

## Why it is safe enough

- **Every call needs a key.** Each person has their own, and the host can remove it.
- **HTTPS with a pinned certificate.** The client only talks to the host's own
  certificate, so nobody in between can read or change the traffic.
- **The API can only do a fixed list of things.** It cannot run commands or read
  files, and it picks voices by id.
- **Only validation voices are offered.** The final test speakers and personal
  custom voices never are.
- **One GPU job at a time, in a queue,** with at most 3 waiting per person.
- **Every call is logged** with who made it, in `history/api/access.log`.

Limits: it works only while the host's PC, the API and the tunnel are running.
The cloned audio of real people stays within the group.

## If something goes wrong

| Message | What to do |
|---|---|
| `STOP: the host's certificate is not the one you trusted` | Ask the host whether they made a new certificate. If they did, run `connect --forget` and connect again. If they did not, do not connect. |
| `The host does not accept this key` | Ask the host for a new key, then run `connect --forget` and connect again. |
| `The host cannot be reached` | The host's API or the playit tunnel is not running. |
| `You already have 3 jobs waiting` | Wait for one of them to finish. |
