"""agreement.py | The Dramatizer user agreement: the text every account accepts before using the site.

The website shows it when someone registers (they must tick that they agree), and to every
account that has not accepted the current VERSION yet (accounts made by the host, and everyone
after a change) at their next login: the API refuses that account everything else until it
accepts. What each account accepted, and when, is kept in the database (storage.py, the
agreements table; rows are only added).

Changing the text: change VERSION too (the date of the change), so everyone is asked again.
This text was drafted for a school project and has not been reviewed by a lawyer.
"""

VERSION = "2026-11-25"       # 2026-11-25: section 3, impersonation (everyone accepts again)
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
    ("3. No impersonation: only voices you have the right to use",
     "Dramatizer copies voices, so it can be misused to pretend to be someone. You agree that:\n"
     "- You upload and use only your own voice, or the voice of a person who has given you clear permission "
     "to clone it for what you are making. Never the voice of anyone else, including famous people, public "
     "figures, family members, colleagues or people who have died, without that permission.\n"
     "- You never use speech made with Dramatizer to pretend to be a real person: not in calls, voice messages, "
     "videos or posts, and not to get money, passwords, personal information or access from anyone (for "
     "example a fake call from a relative, a manager or a bank).\n"
     "- When you share speech made with Dramatizer, you say that it was made with AI, unless it is plainly "
     "fiction that no one could take for a real recording.\n"
     "- You do not use it to make anyone appear to say something they did not say, such as false statements, "
     "insults, threats, or political or sexual content.\n"
     "Every voice upload and every render is recorded with the account that made it. The team may remove a "
     "voice, disable an account, and give these records to the person affected or to the police when a voice "
     "is used to impersonate someone."),
    ("4. You are responsible for how you use Dramatizer",
     "You alone are responsible for how you use Dramatizer and for everything you make with it. You must not use "
     "it to infringe copyright or other intellectual property, or to impersonate, deceive, defraud, harass, "
     "defame or otherwise harm any person, including by copying someone's voice without their consent. If you "
     "use Dramatizer for such acts, you accept full responsibility and liability for them: you cannot deny that "
     "liability or pass it on to the Dramatizer development team, and you agree to cover the team's costs from "
     "any claim that arises from your use."),
    ("5. Limits on use",
     "Dramatizer runs on one computer with one graphics card, shared by everyone. So that no one person can "
     "fill it, every account except the administrators' may make at most 5 renders a minute (lines spoken, "
     "with or without a genome) and ask at most 10 questions a minute in Ask the Dramatizer, and may have at "
     "most 3 jobs waiting at once. Above a limit the site refuses for a moment and says how many seconds to "
     "wait; nothing is lost. The team may change these limits to keep the service working for everyone, and "
     "changes are shown here. Do not try to get around them, for example with several accounts or a script: "
     "the team may disable accounts that do."),
    ("6. If the agreement is broken",
     "The team may disable your account and remove what you made with Dramatizer, and may report misuse to the "
     "people affected or to the authorities."),
    ("7. Changes",
     "If this agreement changes, you are asked to accept the new version before you can use Dramatizer again."),
]


def as_json():
    return {"version": VERSION, "title": TITLE, "sections": [{"heading": h, "text": t} for h, t in SECTIONS]}


def as_text():
    return f"{TITLE} (version {VERSION})\n\n" + "\n\n".join(f"{h}\n{t}" for h, t in SECTIONS)
