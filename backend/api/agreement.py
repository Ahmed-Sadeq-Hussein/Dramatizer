"""agreement.py | The Dramatizer user agreement: the text every account accepts before using the site.

The website shows it when someone registers (they must tick that they agree), and to every
account that has not accepted the current VERSION yet (accounts made by the host, and everyone
after a change) at their next login: the API refuses that account everything else until it
accepts. What each account accepted, and when, is kept in the database (storage.py, the
agreements table; rows are only added).

Changing the text: change VERSION too (the date of the change), so everyone is asked again.
This text was drafted for a school project and has not been reviewed by a lawyer.
"""

VERSION = "2026-11-20"
TITLE = "Dramatizer user agreement"

SECTIONS = [
    ("1. Genomes belong to the Dramatizer development team",
     "Every genome in Dramatizer (the recipes of laughs, breaths, pauses and other sounds placed in a line), "
     "including any genome you start, edit, name, rate or render with, and all rights in them, belong to the "
     "Dramatizer development team. By creating or changing a genome you give the team any rights you might "
     "have in it. You may not claim a genome as yours or take genomes out of Dramatizer to use elsewhere "
     "without the team's written permission."),
    ("2. Your voice is stored, and is not used to train AI",
     "Voice recordings you upload, and the speech made from them, are stored on the Dramatizer host's computer. "
     "They are used only to make speech for your account. They are not shared with other users, and they will "
     "not be used to train, fine-tune or evaluate any AI model or dataset. The host's administrators can see them "
     "in order to run and protect the service. You can delete a voice you uploaded at any time, and it is then "
     "deleted from the host. Upload only your own voice, or the voice of someone who has given you permission "
     "to do so."),
    ("3. You are responsible for how you use Dramatizer",
     "You alone are responsible for how you use Dramatizer and for everything you make with it. You must not use "
     "it to infringe copyright or other intellectual property, or to impersonate, deceive, defraud, harass, "
     "defame or otherwise harm any person, including by copying someone's voice without their consent. If you "
     "use Dramatizer for such acts, you accept full responsibility and liability for them: you cannot deny that "
     "liability or pass it on to the Dramatizer development team, and you agree to cover the team's costs from "
     "any claim that arises from your use."),
    ("4. If the agreement is broken",
     "The team may disable your account and remove what you made with Dramatizer, and may report misuse to the "
     "people affected or to the authorities."),
    ("5. Changes",
     "If this agreement changes, you are asked to accept the new version before you can use Dramatizer again."),
]


def as_json():
    return {"version": VERSION, "title": TITLE, "sections": [{"heading": h, "text": t} for h, t in SECTIONS]}


def as_text():
    return f"{TITLE} (version {VERSION})\n\n" + "\n\n".join(f"{h}\n{t}" for h, t in SECTIONS)
