"""about.py | What the Dramatizer is, how to use it, and who made it: one text for the Home page and the chatbot.

GET /api/v1/about serves ABOUT; the page shows it under "About" on Home, and the chatbot's
about_dramatizer tool reads the same, so the two never disagree.
"""

ABOUT = {
    "what": ("The Dramatizer clones a voice from a short recording and gives a line a delivery: stress on a word, "
             "a pause, a sigh, a breath, a laugh or a change of speed. That set of edits is called a genome. "
             "Listeners score each genome's clip from 0 to 5 against the plain line, and the project studies "
             "which genomes make a line sound better."),
    "how": [
        {"step": "Rate", "text": "Open Rate, choose one of your piles, listen to each clip and give it a score from "
                                 "0 to 5. Score the plain line too: every genome is compared with it."},
        {"step": "Generate", "text": "Open Generate, pick a voice, then a genome from the library or make your own "
                                     "in the editor, type a line and render it."},
        {"step": "Listen", "text": "Open Listen to hear the voices in the data, or the lines people have rendered."},
        {"step": "Ask", "text": "Type a question in Ask the Dramatizer on this page, for example \"Which genomes "
                                "work best for each voice?\". Answers come from the project's own records and "
                                "name them; when nothing answers a question, it says so."},
        {"step": "Research", "text": "Researchers and admins also see the genome searches, their rankings and the "
                                     "measured results."},
    ],
    "contributors": ["Ahmed Hussein", "Julius Norén", "Rama Muharam", "Tran Hieu Minh"],
}


def as_text():
    """The same, as plain text for a language model."""
    steps = "\n".join(f"- {s['step']}: {s['text']}" for s in ABOUT["how"])
    return f"{ABOUT['what']}\n\nHow to use it:\n{steps}\n\nContributors: {', '.join(ABOUT['contributors'])}."
