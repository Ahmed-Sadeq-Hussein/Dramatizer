"""glossary.py | What the Dramatizer's words mean: its measurements and its core ideas, in one place.

    from backend import glossary

    glossary.lookup("what does STOI measure?")     # [{"term": "STOI", "meaning": ..., ...}] best first
    glossary.term("wer")                           # one entry by its name or another name for it, or None
    glossary.entries()                             # every entry

Each entry: term, also (other names), kind (measurement, idea, gene, data, website), meaning (one or two
sentences), scale (how to read its number, for a measurement), details, and where (where the program uses it).

The numbers in it (limits, ranges, counts) are read from the code that uses them, so the glossary says what the
program actually does. The chatbot looks words up here (GET /glossary, its "define" tool), and so does the
Ask tab's answer from the records when the model is not ready.
"""
import re

from backend.genome import artifact_check, evidence_snapshot, genome_factory, ratings
from backend.metrics.reward import CONFIG

_SANITY_WER = round(CONFIG.sanity_wer_limit * 100)

# Each gene type: what it does to the line
GENES = {
    "emphasis": "stresses one word (the word is marked <strong> for the voice model)",
    "accent": "an [accent] tag on a word: a change of stress or intonation there",
    "breath": "an audible breath before a word",
    "quick_breath": "a short, quick breath before a word",
    "sigh": "a sigh at a word",
    "laughter": "a laugh at a word",
    "laughing_speech": "a stretch of words said while laughing",
    "cough": "a cough at a word",
    "clucking": "a tongue click at a word",
    "lipsmack": "a lip smack at a word",
    "mn": "a hesitation hum ('mm', 'mn') at a word",
    "noise": "a non-speech noise at a word",
    "hissing": "a hiss at a word (switched off: new genomes never use it)",
    "vocalized_noise": "a voiced non-word sound at a word (switched off: new genomes never use it)",
    "pause": "a pause after a word: 'dramatic' (written ...) or 'realization' (a sudden stop, written as a dash)",
    "speed": (f"the speaking rate of the whole line, from {genome_factory.SPEED_BOUNDS[0]} (slower) to "
              f"{genome_factory.SPEED_BOUNDS[1]} (faster); 1.0 is unchanged"),
}

ENTRIES = [
    # ------------------------------------------------------------------ measurements --
    {"term": "WER", "also": ["word error rate", "word errors", "errors"], "kind": "measurement",
     "meaning": "Word error rate: how many words of the script a speech recogniser (Whisper) gets wrong when it "
                "listens to a recording. Wrong means replaced, missing or extra, divided by the number of words in "
                "the script.",
     "scale": "0 is perfect; 0.10 means 1 word in 10 wrong; it can go above 1 when many extra words are heard. "
              "Lower is better. The project treats above 0.10 as too many (the Lab marks it).",
     "details": "Fillers such as 'uh', 'hmm' or 'haha' are left out before comparing, because a laugh or hum gene "
                "is often written down as one. A high WER on an original recording usually means its stored "
                "script does not match what is said, or that Whisper misheard a word.",
     "where": "Baselines, the sanity check, the rewards, and the Lab tab (with what was heard, word by word)."},
    {"term": "Speaker similarity", "also": ["similarity", "voice similarity", "resemblyzer", "cosine similarity",
                                            "similarity to reference", "speaker embedding"], "kind": "measurement",
     "meaning": "How much two recordings sound like the same person. Each voice is turned into a 'fingerprint' of "
                "256 numbers (Resemblyzer's speaker encoder) and the two fingerprints are compared (cosine "
                "similarity).",
     "scale": "1.0 is identical; about 0.75 or more sounds like the same person; about 0.5 is different people. "
              "Higher is better.",
     "details": f"The sanity check wants a genome's clip at least {CONFIG.sanity_similarity} similar to its plain "
                f"line ({CONFIG.sanity_similarity_short} when the line is under {CONFIG.sanity_long_seconds:g} s of "
                "speech, because the encoder is unsteady on little speech). It says nothing about how natural the "
                "voice sounds.",
     "where": "Baselines, the sanity check, the voice map, and the Lab tab (to a reference and between every pair)."},
    {"term": "PESQ", "also": ["perceptual evaluation of speech quality"], "kind": "measurement",
     "meaning": "Perceptual Evaluation of Speech Quality: an estimate of how clean a recording sounds (noise, hiss, "
                "distortion, muffling), made from the recording alone by the SQUIM model.",
     "scale": "1.0 to 4.5. About 4 or more is clean; below 2.5 is noisy or distorted (the Lab marks it). Higher is "
              "better.",
     "details": "An estimate, not the original PESQ test, which needs a clean copy to compare with. It does not "
                "judge whether the voice is natural or the right person.",
     "where": "Baselines, the rewards and the Lab tab."},
    {"term": "STOI", "also": ["short-time objective intelligibility", "intelligibility"], "kind": "measurement",
     "meaning": "Short-Time Objective Intelligibility: an estimate of how easy the speech is to understand, made "
                "from the recording alone by the SQUIM model. It looks at how well the short-time patterns of "
                "speech survive noise and distortion.",
     "scale": "0 to 1. Close to 1 (0.95 or more) is very clear; lower means words get harder to make out. Higher "
              "is better.",
     "details": "It measures whether the speech can be understood, not whether the right words are said: that is "
                "WER.",
     "where": "Baselines and the Lab tab."},
    {"term": "SI-SDR", "also": ["si sdr", "sisdr", "signal to distortion ratio", "scale-invariant signal-to-distortion ratio"],
     "kind": "measurement",
     "meaning": "Scale-invariant signal-to-distortion ratio: an estimate, in decibels, of how much of the recording "
                "is clean speech against noise and distortion, made from the recording alone by the SQUIM model.",
     "scale": "In dB; higher is cleaner. Around 20 dB or more is clean; near 0 dB the noise is as loud as the voice.",
     "details": "'Scale-invariant' means turning the volume up or down does not change it.",
     "where": "Baselines and the Lab tab."},
    {"term": "SQUIM", "also": ["quality", "perceptual quality", "perq", "quality model", "squim"], "kind": "measurement",
     "meaning": "The model (torchaudio's SQUIM) that estimates PESQ, STOI and SI-SDR from one recording, without "
                "a clean reference to compare with.",
     "scale": "See PESQ, STOI and SI-SDR.",
     "details": "These estimate how clean the signal is. They do not judge whether the voice sounds natural or "
                "human, or like the right person: similarity and listeners do.",
     "where": "backend/metrics/PERQ.py; PERQ_MODEL in .env."},
    {"term": "Speech seconds", "also": ["speech s", "speaking time", "speech time", "seconds"], "kind": "measurement",
     "meaning": "How long a recording actually speaks: its length with the silences and pauses taken out. "
                "'Seconds' is the whole length.",
     "scale": "In seconds. The difference between the two is the time spent in pauses.",
     "details": "Used for the speaking rate and the pace, so a long pause does not count as slow speech.",
     "where": "The Lab tab and the sanity check."},
    {"term": "Words per second", "also": ["words/s", "speaking rate", "rate"], "kind": "measurement",
     "meaning": "The speaking rate: the words of the script divided by the speech seconds (pauses left out).",
     "scale": "Normal speech is roughly 2.5 to 3.5 words per second. Lower is slower.",
     "details": "Comparing a render's rate with its original's shows whether a genome (or the voice model) speaks "
                "slower or faster than the person.",
     "where": "The Lab tab and the measurements of clones."},
    {"term": "Pace", "also": ["pace limit"], "kind": "measurement",
     "meaning": "In the sanity check: the plain line's speaking time divided by the genome clip's, after allowing for "
                "the speed the genome asked for.",
     "scale": f"1.0 is as fast as it should be; {CONFIG.sanity_pace_min} means 30% slower. Below "
              f"{CONFIG.sanity_pace_min} the clip fails.",
     "details": "It stops a clip from getting its words right by dragging them out.",
     "where": "The sanity check."},
    {"term": "MOS", "also": ["mean opinion score", "score", "scores", "rating", "ratings", "listener score"],
     "kind": "measurement",
     "meaning": "Mean opinion score: the average of the listeners' scores of a clip. Each listener scores a clip "
                f"from {ratings.LOWEST:g} (bad) to {ratings.HIGHEST:g} (excellent).",
     "scale": f"{ratings.LOWEST:g} to {ratings.HIGHEST:g}; higher is better. A score from one listener is that "
              "listener's opinion, not a result.",
     "details": evidence_snapshot.RULE,
     "where": "The Rate page, the rankings, the evidence and the graphs."},
    {"term": "Above plain", "also": ["above the plain line", "above_plain", "mean above plain"], "kind": "measurement",
     "meaning": "A genome's score minus the same listener's score for the plain line of that voice: how much the "
                "genome improved (or spoiled) the line for that listener.",
     "scale": "Positive is better than the plain line, 0 the same, negative worse.",
     "details": "Taking each listener's own plain-line score away removes how strict or generous that listener is.",
     "where": "The evidence, the rankings and the research questions."},
    {"term": "Beat the plain line", "also": ["beat_plain", "beats the plain line", "share better than the plain line"],
     "kind": "measurement",
     "meaning": "The share of a group's genome scores that are above the same listener's plain-line score.",
     "scale": "0 to 1; 0.26 means 26% of the scores beat the plain line.",
     "details": "Research question 5 uses it per number of genes.",
     "where": "The evidence (query_scores grouped) and question 5."},
    {"term": "Confidence interval", "also": ["interval", "95% interval", "confidence range", "ci"], "kind": "measurement",
     "meaning": "The range a mean probably lies in, given how much the listeners disagree and how many there are.",
     "scale": f"A {round(evidence_snapshot.CONFIDENCE * 100)}% t-interval over the listeners' scores. With one "
              "listener there is none.",
     "details": evidence_snapshot.WINNER_RULE,
     "where": "The evidence snapshot (winners) and the baselines."},
    {"term": "Sanity check", "also": ["sanity", "verdict", "fine", "terrible", "sanity verdict"], "kind": "measurement",
     "meaning": "The automatic check of each genome's clip against the plain line of the same line and voice. "
                "Its verdict is 'fine' or 'terrible'.",
     "scale": f"'fine' only if all three hold: (1) what is heard differs from the plain line by at most "
              f"{_SANITY_WER}% of the words (at least {CONFIG.sanity_wer_words} word may differ; fillers do not "
              f"count), (2) it still sounds like the same speaker (similarity to the plain line at least "
              f"{CONFIG.sanity_similarity}, or {CONFIG.sanity_similarity_short} on short lines), and (3) its pace is "
              f"at least {CONFIG.sanity_pace_min}. Any one failing makes it 'terrible'.",
     "details": "The limits were calibrated on 400 pairs of plain renders (6 Nov 2026), so ordinary variation "
                "between renders passes.",
     "where": "Every genome search; the verdict filter of the scores."},
    {"term": "Artifact check", "also": ["artifact", "artifacts", "clipping", "burst", "gap"], "kind": "measurement",
     "meaning": "An automatic check of a rendered clip's audio for faults a listener would hear, before its score is "
                "used to recommend it.",
     "scale": f"Fails on: clipping (more than {artifact_check.CLIP_SHARE * 100:g}% of samples at full scale), a "
              f"burst (its loudest 20 ms more than {artifact_check.BURST_DB:g} dB louder than the plain line's), a "
              f"gap (silence longer than {artifact_check.GAP_SECONDS:g} s), too short (under "
              f"{artifact_check.MIN_VOICED_SECONDS:g} s of sound), or an unreadable file.",
     "details": "The sanity check asks whether the words, voice and pace survived; this one hears clicks, bursts "
                "and gaps. Listeners can also tick that they heard an artifact.",
     "where": "Research question 2 and the evidence snapshot."},
    {"term": "Different person", "also": ["different speaker", "sounded like a different person"], "kind": "measurement",
     "meaning": "A tick box on the Rate page: the listener heard someone other than the voice's original recording.",
     "scale": "Yes or no per clip and listener.",
     "details": "Similarity measures the same thing by machine; this is the listener's own judgement.",
     "where": "The Rate page and the evidence."},
    {"term": "Reward", "also": ["rewards", "wer reward", "similarity reward", "quality reward"], "kind": "measurement",
     "meaning": "Turning WER, similarity and quality into a 0-to-1 score each, so clips can be ranked automatically.",
     "scale": f"1 is full marks, 0 none, a straight line between. WER: full up to {CONFIG.wer_free}, none at "
              f"{CONFIG.wer_free + CONFIG.wer_fade:g}. Similarity: full at {CONFIG.sim_full}, none at "
              f"{CONFIG.sim_zero}.",
     "details": "Listeners' scores, not rewards, decide which genomes are best.",
     "where": "backend/metrics/reward.py."},

    # ----------------------------------------------------------------------- ideas --
    {"term": "Dramatizer", "also": ["the dramatizer", "this program", "the project"], "kind": "idea",
     "meaning": "A project that clones a person's voice from a short recording and gives a line a delivery: "
                "sighs, pauses, laughs, stress and speed, written as a genome. Listeners rate the results, and the "
                "ratings show which deliveries work.",
     "scale": "", "details": "", "where": "The whole site; ask 'how do I use the site' for its tabs."},
    {"term": "Genome", "also": ["genomes", "delivery"], "kind": "idea",
     "meaning": "A delivery for a line: a short list of genes (edits) such as a sigh at one word, a pause after "
                "another, or a speed change. The same genome fits any line, because each gene's place is given as a "
                "position from the first word to the last.",
     "scale": f"A random genome has {genome_factory.MIN_OPERATIONS} to {genome_factory.MAX_OPERATIONS} genes; one "
              "made by hand up to 10.",
     "details": "A genome is numbered within its search (genome 1 to 10 of a voice), may be named, or made by hand.",
     "where": "Searches, the studio (Line and genome), ratings and the evidence."},
    {"term": "Gene", "also": ["genes", "gene type", "gene types", "feature", "features", "active features"],
     "kind": "idea",
     "meaning": "One edit in a genome: a type (sigh, pause, laughter, speed...), where in the line it goes, and for "
                "some types a setting. The types are: " + "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in GENES.items()) + ".",
     "scale": "Position runs from 0 (the first word) to 1 (the last).",
     "details": "Hissing and vocalized noise are switched off: old genomes keep them, new ones never use them.",
     "where": "Searches, the studio and the evidence's gene-type groups."},
    {"term": "Plain line", "also": ["plain", "baseline clip", "neutral render", "no genome"], "kind": "idea",
     "meaning": "The line rendered in the voice with no genome at all. Every genome is judged against the plain "
                "line of the same line and voice.",
     "scale": "", "details": "On the Rate page it is mixed in blind among the genomes, so it is scored the same way.",
     "where": "Searches (baseline.wav), the Rate page and 'above plain'."},
    {"term": "Baseline", "also": ["baselines", "baseline results"], "kind": "idea",
     "meaning": "Two uses: (1) in a genome search, the plain line it renders first; (2) the measured baselines: plain "
                "voice cloning of validation voices with no genome, measured (WER, similarity, PESQ, STOI, SI-SDR) "
                "to show how good cloning is before any genome.",
     "scale": "", "details": "", "where": "Searches; the Results tab and baseline_results."},
    {"term": "Reference recording", "also": ["reference", "original", "original recording", "reference audio"],
     "kind": "idea",
     "meaning": "The real recording of a person that the voice model clones from, with its script. In the Lab, the "
                "reference is the recording the others are compared with for similarity.",
     "scale": "", "details": "A voice's script is needed: the model hears the recording and its words together.",
     "where": "Voices, searches (played first on the Rate page) and the Lab."},
    {"term": "Voice cloning", "also": ["clone", "cloning", "cosyvoice", "voice model", "tts", "text to speech"],
     "kind": "idea",
     "meaning": "Making new speech in a person's voice from a short recording of them. The Dramatizer uses "
                "CosyVoice 3, which takes the reference recording, its script and the new line (with the genome's "
                "tags) and speaks it.",
     "scale": "", "details": "", "where": "Every render and search."},
    {"term": "Render", "also": ["renders", "rendered", "generate", "speak"], "kind": "idea",
     "meaning": "A line spoken in a voice by the voice model, with or without a genome. Renders made on the site are "
                "kept with who made them, the line, the genome and the seed.",
     "scale": "", "details": "Admins can delete renders; they are moved aside, not erased.",
     "where": "Generate (makes them), Listen > Renders (plays them) and the Lab (measures them)."},
    {"term": "Seed", "also": ["render seed", "random seed"], "kind": "idea",
     "meaning": "A number that fixes the voice model's randomness, so the same seed, voice, line and genome give the "
                "same clip. In a search every clip uses the same render seed, so clips differ only by their genome.",
     "scale": "", "details": "A genome seed separately decides which random genomes a search draws.",
     "where": "Renders and searches."},
    {"term": "Markup", "also": ["tags", "tagged line"], "kind": "idea",
     "meaning": "The line as the voice model receives it, with the genome written in: tags such as [sigh] or "
                "[laughter], <strong> around a stressed word, '...' for a dramatic pause.",
     "scale": "", "details": "", "where": "The studio's preview, renders and searches."},
    {"term": "Genome search", "also": ["search", "searches", "run", "runs", "random search", "guided search",
                                       "preference guidance"], "kind": "idea",
     "meaning": "Trying genomes on one line in one voice: render the plain line, render 10 genomes, and judge each "
                "with the sanity check. 'Random search' draws random genomes; the guided methods rank mutations of "
                "a starting genome with a preference model trained on listeners' ratings.",
     "scale": "", "details": "A search's clips can then be put in a listening pile for people to rate.",
     "where": "Research > Searches."},
    {"term": "Named genome", "also": ["named genomes", "pick", "picked genome", "made by hand", "edited genome",
                                      "private genome", "my genomes"], "kind": "idea",
     "meaning": "A genome given a name so it can be used again: picked from a search, or made by hand in the studio. "
                "A genome made by hand is seen only by the account that made it.",
     "scale": "", "details": "", "where": "The studio's genome list."},
    {"term": "Listening pile", "also": ["pile", "piles", "batch", "batches", "listening batch"], "kind": "idea",
     "meaning": "A set of clips (several voices' searches) for one listener to score. A batch is a group of piles "
                "made for one listening study, such as mos-diverse-v1 or policy-vs-random-v1.",
     "scale": "", "details": "Accounts are assigned the piles they should rate.",
     "where": "The Rate page."},
    {"term": "Blind", "also": ["blind rating", "blind pile", "sealed", "report evaluation"], "kind": "idea",
     "meaning": "A blind pile names its clips by tokens and shuffles them per listener, so nobody can tell the plain "
                "line or a method from the others, and listeners do not see each other's scores. A sealed study "
                "(the Report evaluation) keeps which version is which in a key that is opened only when every score "
                "is in.",
     "scale": "", "details": "", "where": "The Rate page."},
    {"term": "Voice", "also": ["voices", "speaker", "speakers", "dataset", "datasets", "vctk", "crema-d", "libritts-r"],
     "kind": "data",
     "meaning": "A person's voice from one of the datasets, written DATASET/SPEAKER (for example VCTK/p250, "
                "CREMA-D/1014, LibriTTS-R/6319). VCTK is read English speech, CREMA-D acted emotional speech, "
                "LibriTTS-R audiobook speech.",
     "scale": "", "details": "Ask 'how many voices are there' for the counts.",
     "where": "Generate > Voices and the voice map."},
    {"term": "Validation and test voices", "also": ["validation", "test", "locked test", "split", "test voices"],
     "kind": "data",
     "meaning": "The voices are split once: validation voices are used to search and tune; test voices are locked "
                "away and used only for the final evaluation, so the results are not tuned to them.",
     "scale": "", "details": "Only validation voices are offered on the site.",
     "where": "The voice manifest and its split."},
    {"term": "Custom voice", "also": ["custom voices", "uploaded voice", "uploaded voices", "my voice", "upload"],
     "kind": "data",
     "meaning": "A person's own recorded voice: set up by the host, or uploaded by an account (3 to 30 s, at most "
                "3 per account). Only its owner and admins can use it, and it is never used for genome searches.",
     "scale": "", "details": "Uploaded voices are stored but not used to train AI (the user agreement).",
     "where": "Generate > Voices > Your uploaded voices."},
    {"term": "Evidence", "also": ["evidence record", "records", "snapshot", "evidence snapshot", "winner",
                                  "research questions", "six questions"], "kind": "idea",
     "meaning": "The project's records of the listeners' scores, searches and sanity checks, which the charts and "
                "the chatbot answer from. The snapshot is the analysis at one moment: coverage, per voice a winner "
                "only when the evidence allows one, the artifact checks and caveats. The six research questions are "
                "answered from it.",
     "scale": "", "details": evidence_snapshot.WINNER_RULE,
     "where": "Research > Graphs and the Ask tab."},
    {"term": "Milestone 2", "also": ["milestone"], "kind": "idea",
     "meaning": "The course milestone that ran random genome searches on the baseline's voices and lines, and "
                "counted how many genomes passed the checks against the baseline, and why the others failed.",
     "scale": "", "details": "", "where": "The Results tab."},
    {"term": "Lab", "also": ["lab tab", "compare recordings"], "kind": "website",
     "meaning": "The tab that measures up to 8 recordings side by side (renders, original voices, search clips): "
                "WER with what was heard word by word, speaker similarity to a reference and between every pair, "
                "PESQ, STOI, SI-SDR, length and speaking rate.",
     "scale": "", "details": "", "where": "The Lab mode in the top bar."},
]

for _name, _what in GENES.items():
    ENTRIES.append({"term": _name.replace("_", " ").capitalize() + " (gene)", "also": [_name, _name.replace("_", " ")],
                    "kind": "gene", "meaning": f"A gene that adds {_what}." if not _what.startswith(("the ", "a stretch"))
                    else f"A gene: {_what}.", "scale": "", "details": "", "where": "Genomes."})

STOP = {"what", "whats", "is", "are", "the", "a", "an", "of", "does", "do", "it", "mean", "means", "measure",
        "measures", "measured", "define", "definition", "explain", "meaning", "in", "this", "program", "dramatizer",
        "how", "by", "and", "or", "for", "to", "specifically", "exactly", "but", "me", "tell", "about", "you", "can",
        "really", "again", "more", "detail", "details", "please", "so", "then", "that", "its", "they", "them",
        "those", "these", "there", "here", "used", "use", "work", "works", "stand", "stands", "word", "term",
        "called", "thing", "means", "say", "said", "on", "at", "with", "one", "your", "we", "our", "i"}
DEFINING = re.compile(r"^\s*(but\s+|and\s+|so\s+)?(what('?s| is| are| does| do)|define|definition|meaning|explain|"
                      r"tell me (what|about)|what do you mean|describe)\b|\b(stand for|mean|means|measure|measures|"
                      r"measured)\s*\??\s*$", re.IGNORECASE)


def _norm(text):
    return re.sub(r"[^a-z0-9/ -]+", " ", str(text or "").lower().replace("_", " ")).strip()


def _names(entry):
    return [_norm(entry["term"]), *(_norm(a) for a in entry["also"])]


def entries():
    return list(ENTRIES)


def term(name):
    """One entry by its name or another name for it (any case), or None."""
    wanted = _norm(name)
    return next((e for e in ENTRIES if wanted in _names(e)), None)


def lookup(question, limit=3):
    """The entries a question asks about, best first: a name said in full (longest first), then shared words."""
    text = f" {_norm(question)} "
    scored = []
    for entry in ENTRIES:
        whole = max((len(n) for n in _names(entry) if n and f" {n} " in text), default=0)
        if whole:
            scored.append((2, whole, entry))
            continue
        words = {w for w in text.split() if w not in STOP and len(w) > 2}
        shared = words & set(" ".join(_names(entry)).split())
        if shared:
            scored.append((1, len(shared), entry))
    scored.sort(key=lambda s: (-s[0], -s[1], s[2]["kind"] == "gene"))
    return [e for _, _, e in scored[:limit]]


def best(question):
    """The entry a question names outright (its name or another name, said in full), or None."""
    found = lookup(question, limit=1)
    if not found:
        return None
    text = f" {_norm(question)} "
    return found[0] if any(n and f" {n} " in text for n in _names(found[0])) else None


def as_text(entry):
    parts = [f"{entry['term']}: {entry['meaning']}"]
    for key, label in (("scale", "How to read it"), ("details", "Also"), ("where", "Where it is used")):
        if entry.get(key):
            parts.append(f"{label}: {entry[key]}")
    return "\n".join(parts)


def asked(question):
    """
    For a question that only asks what something means ("what is STOI?", "explain the sanity check"):
    (entry, None) when it names a term and nothing else; (None, "follow-up") when it names nothing ("but what
    does it measure?": the term is in the conversation); (None, None) for any other question.
    """
    if not DEFINING.search(question or ""):
        return None, None
    words = [w for w in _norm(question).split() if w not in STOP and len(w) > 1]
    if not words:
        return None, "follow-up"
    entry = best(question)
    if entry and set(words) <= set(" ".join(_names(entry)).split()) | {"gene", "genes"}:
        return entry, None
    return None, None
