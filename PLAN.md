# Dramatizer 2: plan

Goal: match everything the first Dramatizer does, fix what it got wrong, and
turn the genome search into a result strong enough to publish.

Written 26 October 2026. Citations marked (verify) come from memory and must
be checked against the papers before they go into anything we submit.

A note on words: the method is a **genetic (evolutionary) search over
genomes**, not "genome sequencing". Sequencing means reading DNA. Use
"genome", "search" or "evolution" in the paper.

---

## 1. Where we start

**Dramatizer 1 does (the bar to match):**

| Part | What it does |
|---|---|
| Markup (`tts_formatter.py`) | Option table: emphasis, pauses, event tokens, speed, instructions; clamping, warnings, strict mode |
| Genome (`candidate.py`, `genome_factory.py`) | Genes hold a *position* in the sentence (0 to 1), not a word, so a genome transfers between voices and lines. Four mutations: add, remove, modify, replace |
| Search (`explorer.py`) | Mutate random parents, accept by gates, population cap, duplicate skipping |
| Reward (`unsupervised_measurements.py`) | validity x intelligibility x same speaker x expressiveness (Wundt curve on novelty) |
| Baseline (`dramatizer.py`) | Neutral render (best of 3), render retries when the length is wrong, optional seed |
| Memory (`genome_history.py`) | Winner of the nearest earlier voice (speaker embedding) seeds the next voice |
| Human loop (`voice_cli.py`) | Listen, rate 1 to 5, pick a winner |
| Records | `run.json` log per run, ratings, history; demo mode without a model |
| Voice map (`voice_map.py`) | Timbre and style maps, clustering |

**Dramatizer 1 gets wrong (fix, do not copy):**

1. Novelty ignores earlier candidates (no archive passed to `voice_reward`).
2. Three of six validity checks never run (emphasis sense, pause sense; parsimony threshold above the operation cap).
3. The reward we optimise is also the number we report. That is circular.
4. A missing recogniser breaks every candidate, but not the baseline.
5. Two writers for the history file, with different record formats.
6. The word tokenizer is copied in five places.
7. No tests. Audio and results were committed to git on one branch.
8. Voices are copyrighted characters: not publishable, and a responsible-use problem.

**Dramatizer 2 has so far:** CosyVoice 3 installed and working, `Voice_model.py`
(`generate_speech`), `Voice_Importer.py` and `Voice_UI.py` over CREMA-D and
VCTK, `.env`, `requirements.txt`, `Voice library.txt`.

---

## 2. Build order

Each phase ends with something we can run and show. Do not start the next
before the check passes.

### Phase 1: the model layer (finish what exists)
- Backend interface: `Backend` with `generate_speech`, `features`, `to_markup`. CosyVoice 3 is the first implementation; a `demo` backend needs no GPU.
- **Re-verify the event tokens for CosyVoice 3 by listening.** The old list (`[breath]`, `[laughter]`, ...) was checked on CosyVoice 2. Check the tokenizer file in the checkout for the supported set, render each token on 3 voices, keep only the ones that are audible. This decides what the genome may use.
- Check: every token has a verdict (works, weak, absent) and an audio example.

### Phase 2: the script and markup (replace `tts_formatter.py`)
- `Script`: words with marks on word indexes, written to markup once at the end. This removes the regex machinery and the tokenizer bug (one `words()` function).
- Port the option table, bounds, warnings and strict mode with their tests.
- Check: for every genome in the 42 old runs, the new markup equals the old markup (or the difference is explained).

### Phase 3: genome and search
- `Candidate`, factory and mutation operators ported (section 3 lists upgrades to try after parity).
- One history module, one record format with a `schema_version`.
- Check: a 30-step run in demo mode produces the same kind of `run.json`.

### Phase 4: measurement, redesigned
- Same reward terms, but fix items 1 and 2 above.
- Separate **search reward** from **evaluation metrics**: search uses Whisper-base WER, Resemblyzer similarity and the Wundt term; reporting uses *different* models (another recogniser, e.g. a larger Whisper or Parakeet; another speaker encoder, e.g. ECAPA-TDNN or WavLM-based; a predicted-quality model such as UTMOS; emotion2vec for expressiveness) plus humans (verify each tool before use).
- Load audio in Python for Whisper (it needs ffmpeg otherwise on Windows).
- Check: swapping the judge changes numbers, not conclusions (section 5).

### Phase 5: review tools
- Flask UI (`Voice_UI.py` exists) gets a review page: play the shortlist, rate, pick a winner, blind A/B mode for the listening test.
- Voice map ported; run on the new library (VCTK, CREMA-D) so clusters reflect voices, not recording sources.

### Phase 6: tests, docs, CI
- Tests at three levels: unit (markup, genome), integration (demo run end to end), smoke (one real render, skipped without GPU).
- Git: `.gitignore` first (`.env`, audio, output), `.env.example` committed, one branch, a clean-clone check of the README.

---

## 3. Algorithms that could make it better

The current search is a plain mutate-and-filter loop that keeps everything
passing a gate. It has no explicit notion of *different kinds of good*. Ideas,
cheapest and most promising first:

| # | Algorithm | What it fixes | Cost |
|---|---|---|---|
| 1 | **Quality-diversity: MAP-Elites** (Mouret & Clune, 2015, verify); library **pyribs** | Keeps the best clip *per region* of the style space (tempo, pauses, pitch range, loudness) instead of one pool. The 6 behaviour numbers already exist | Low: the descriptors are already measured |
| 2 | **Novelty search with a real archive** (Lehman & Stanley, 2011, verify) | Fixes the missing archive; rewards "new", not just "different from plain" | Very low: pass the archive in |
| 3 | **CMA-ME / CMA-MAE** (Fontaine et al., 2020 and 2022, verify) | Smarter moves inside MAP-Elites for the continuous genes (positions, rates) | Medium (pyribs has it) |
| 4 | **Multi-objective search: NSGA-II** (Deb et al., 2002) | Treats intelligibility, similarity, expressiveness as separate goals; gives a Pareto front instead of one multiplied score | Medium: removes the arbitrary product |
| 5 | **Bandit choice of mutation operator** (UCB or Thompson sampling) | Learns which operator (add, remove, modify, replace) pays off for each voice | Low |
| 6 | **Preference learning from human ratings** (Bradley-Terry; preferential Bayesian optimisation: Chu & Ghahramani 2005; Brochu et al. 2010, verify) | Uses the human picks to learn what *people* like, not just the machine score; pairwise choices are more reliable than 1-5 ratings | Medium; needs a rating sessions |
| 7 | **Surrogate model** (small regressor from genome + line + voice features to reward) | Skips most renders (about 1 s per second of speech): rank thousands of genomes, render the best | Medium; needs data from past runs |
| 8 | **LLM-guided mutation** (Evolution through Large Models, Lehman et al., 2022; LLM-as-crossover, verify) | The model proposes edits that fit the *meaning* of the line ("stress `never`", "pause before the lie") | Medium; also a baseline (below) |
| 9 | **Crossover between genomes** | The current code only mutates. Combine the first half of one winner with the second of another | Low |
| 10 | **Fine-tune with the reward**: DPO or GRPO on the TTS model, using human or learned preferences (CosyVoice 3 itself was improved with RL (verify)) | Moves the improvement *into the model*, so no search is needed at run time | High; future work |

Recommended path: 2 and 1 first (small code, clear story), then 5, then 6.
Items 7 and 8 make strong comparison methods. Item 10 is a follow-up paper.

---

## 4. The paper

**Working claim (narrowed after the related-work survey, section 7).**
"Evolving TTS markup" is already prior art (LAION). Our claim is the
combination: the first quality-diversity search over the inline markup of a
*frozen* zero-shot voice-cloning TTS, with identity, intelligibility and
validity guards and a Wundt-curve novelty objective, a diverse shortlist for
a human pick, and transfer of found deliveries across lines and voices via
position-relative genes and nearest-speaker seeding. Always say "to our
knowledge".

**Contributions to demonstrate:**
1. A position-based, model-independent genome for paralinguistic markup.
2. A reward with guards (intelligibility, speaker identity, validity) and a Wundt-curve expressiveness term.
3. Cross-voice transfer: seeding from the nearest speaker beats a random start.
4. A human evaluation showing the search beats simple alternatives.

**Experiments (what reviewers will ask for):**

| Question | Design |
|---|---|
| Does search beat simple methods? | The seven baselines in section 7, all at the same render budget. Same voices and lines |
| Which parts matter? | Ablations: no Wundt term, no guards, no archive, no transfer, MAP-Elites vs the plain loop |
| Does transfer work? | Nearest-voice seed vs random seed vs a *far* voice's genome, measured in steps to a given score |
| Are the numbers trustworthy? | Report on **different** automatic judges than the search used; human ratings; correlation between judges and humans |
| Does it generalise? | Hold out **speakers** (not clips) and lines. Never let a test speaker enter the history |
| Is the effect real? | Several seeds per condition; bootstrap confidence intervals; Wilcoxon signed-rank for paired comparisons |

**Human study.** 20 or more listeners, blind pairwise preferences (A/B) and
a 1-5 naturalness and expressiveness rating, attention checks, a consent
form, no personal data kept. Follow the ethics rules of the university first.

**Data.** VCTK and CREMA-D (and Expresso / EARS for expressive reference,
non-commercial) are citable and shareable. Do **not** use character voices
from shows or games in the paper's results: not redistributable, and a
consent problem. Our own group's voices need written consent.

**Ethics section (required, not optional).** Zero-shot cloning can be abused.
State consent for every cloned voice, the dataset licences, and what we
release (code, genomes) and withhold (cloned audio of real people).

**Venues:** see section 7. Interspeech 2027 is the main target; EvoMUSART
2027 closes 1 November 2026 (short paper only if the system is ready);
GECCO, SSW and ICCC are alternatives. Verify deadlines before planning.

**Writing plan.** Start the results tables and figures early; the voice map,
cloning-quality plot and run trees already exist in some form. Write the
paper from the experiments, not the other way round.

---

## 5. Rules so the results hold up

- **Freeze a test set now** (speakers x lines x requested styles) and do not look at it until the end.
- **Do not grade with the reward we optimise.** Search and evaluation use different models.
- **Split by speaker.** History must never contain a test speaker.
- **Record everything:** seeds, model and weights versions, `.env` values without secrets, the git commit. A run must be repeatable.
- **Report failures.** Voices that clone badly (high WER) are findings, not noise to hide.

---

## 6. First week

1. `git init`, `.gitignore`, `.env.example`.
2. Verify CosyVoice 3 event tokens by listening (Phase 1).
3. `Script` and markup port, with parity check against old runs (Phase 2).
4. Write the experiment table above into a spreadsheet: voices, lines, seeds, who runs what.
5. Pick the venue and read two recent papers from it.
6. Read the three closest works in section 7 in full, and save Wayback Machine snapshots of the two LAION pages (they are undated).

---

## 7. Related work (survey of 27 October 2026)

Full survey, with every citation, link and a verified/unverified mark:
[`reports/Dramatizer related work survey.md`](reports/Dramatizer%20related%20work%20survey.md).
Raw notes per area: `research_notes/Dramatizer related work survey/`.

**Verdict.** Nothing found does what Dramatizer does end to end, but every
ingredient has prior art, so the novelty is the combination (section 4).

**The three closest works:**

| Work | What it shares | What it lacks |
|---|---|---|
| LAION, MOSS-TTS "emotion evolution" pages (web, undated, no authors) | A genetic algorithm over TTS prompt text and inline vocal-burst tags; emotion fitness minus a WER penalty | No quality-diversity or novelty term, no speaker-similarity guard, no human pick, no transfer; changes the model through LoRA strength |
| Zhou, Atmaja & Sakti, Interspeech 2026 (arXiv 2608.00998) | Interactive GA: listeners pick parents | Genome is a 2-D emotion point, not markup; no automatic scoring; no transfer |
| Song et al., LoopTTS, EMNLP 2026 (arXiv 2608.28970) | An audio-LLM critic proposes word-level stress and pause edits | One correction pass with a trained refiner; no population, no search, no human |

**What each field gives us, in one line:**

1. *Tag and instruction TTS* (CosyVoice 2/3, ElevenLabs v3, Dia, OpenAudio S1): the control surface. CosyVoice 2's own numbers show tags raise expressiveness but slightly worsen WER and similarity, so a guards-only reward drives the search to use no tags.
2. *LLM-written markup*: one-shot only; open LLMs "consistently under-generate tags" (arXiv 2508.17494). Our main baseline.
3. *Search over TTS inputs*: interactive evolution of prosody since 2002 (Sato), always over continuous parameters, and plagued by user fatigue. Our auto-score-then-shortlist answers the fatigue.
4. *Reward tuning of TTS*: documents the failures to avoid. A WER reward makes speech slow and flat (Seed-TTS); GRPO on error rates collapses prosody to monotone (arXiv 2509.18531). Best-of-8 was about as good as RL to listeners (arXiv 2608.31035).
5. *Quality-diversity*: used for sound synthesis (Jónsson et al.) and text prompts, never for speech. Learned behaviour spaces gave more diversity than hand-picked descriptors (arXiv 2512.02783).
6. *Wundt curve*: the original (Saunders & Gero) is asymmetric and measures novelty against a memory that habituates; no study has tested it on listener preference for prosody. Testing it is a contribution.
7. *Evaluation*: every automatic speech metric we use is biased against expressive delivery (MOS predictors favour formal speech; similarity drops with emotion; SQUIM measures cleanliness).
8. *Prosody transfer*: latent prosody transfer does not really transfer (Sigurgeirsson & King, ICASSP 2023), which motivates our symbolic genome. Warm-starting search from the most similar past task works in other fields when the tasks are similar.

**Baselines to beat, all at the same render budget:**

1. Neutral render (best of 3)
2. Guard-only best-of-N over random seeds, no markup
3. Random markup genomes, same guards
4. One-shot markup from a frontier LLM, plus per-line natural-language instructions
5. LLM critique-and-revise loop editing markup on the frozen model
6. LAION-style genetic algorithm maximising one emotion score with a WER penalty
7. Dramatizer without the novelty term

**Design changes the literature asks for** (add to Phase 4 and section 3):

1. Every optimised scorer gets a *different* held-out judge for reporting (e.g. Resemblyzer for search, WavLM-SV and ECAPA for reporting; SQUIM for search, UTMOSv2 for reporting).
2. SQUIM is a threshold guard only; never maximise it.
3. WER and similarity are accepted ranges relative to the neutral render, not things to minimise or maximise.
4. Novelty against an archive of earlier candidates (MAP-Elites over a CVT grid, or Dominated Novelty Search), not only distance from neutral.
5. Ablation: hand-picked 6 descriptors vs a learned PCA behaviour space.
6. An asymmetric Wundt curve (difference of two sigmoids), since the guards already punish going too far.
7. A quality-aware shortlist: the best clip from each of the most distant archive cells.
8. Seed new voices from the top genomes of several nearest voices; log human picks as preference pairs for a later DPO paper.

**Check by hand before submitting:** the list at the end of the full survey
(LAION page dates, a few author lists and venues marked "from memory").
