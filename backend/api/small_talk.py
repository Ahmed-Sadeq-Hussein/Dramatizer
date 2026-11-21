"""small_talk.py | What "Ask the Dramatizer" says to a greeting, and how it tells a Dramatizer question from any other.

Used by the API server (server.py, before the model or the records are asked) and by the chatbot itself
(API calls/chatbot.py). No model and no data: plain patterns, so a greeting is answered at once and a
question about something else ("How many cookies can I eat?") is politely declined instead of being
matched against the records, which once gave a made-up looking answer.

    greeting("hello!")        -> the greeting reply (None when it is not small talk)
    about_us("How many cookies can I eat?")   -> False
    about_us("Did Rama rate any genomes 5?")  -> True
"""
import re

# The whole message is small talk: a greeting, thanks, goodbye, or "who are you / what can you do"
GREETING = re.compile(
    r"^\s*(?:(?:hi|hello|hey|hiya|howdy|yo|greetings|good\s+(?:morning|afternoon|evening|day)|hej|hallo|salut|hola)"
    r"(?:\s+(?:there|again|everyone|all|dramatizer|bot))?"
    r"|how\s+are\s+you(?:\s+doing)?(?:\s+today)?|what'?s\s+up|sup"
    r"|thanks?(?:\s+you)?(?:\s+(?:so|very)\s+much)?|thank\s+you(?:\s+(?:so|very)\s+much)?|cheers|ty"
    r"|bye|goodbye|see\s+you|good\s*night"
    r"|who\s+are\s+you|what\s+are\s+you|what\s+can\s+you\s+do|what\s+do\s+you\s+do)"
    r"[\s,.!?:;)(\-]*(?:dramatizer|bot)?[\s.!?]*$", re.IGNORECASE)
THANKS = re.compile(r"^\s*(thanks?|thank\s+you|cheers|ty)\b", re.IGNORECASE)
BYE = re.compile(r"^\s*(bye|goodbye|see\s+you|good\s*night)\b", re.IGNORECASE)

HELLO_REPLY = ("Hello! I'm the Dramatizer's assistant. I can look things up in the project's records for you: "
               "genomes and how listeners rated them, voices, listening piles, results, and how to use this site. "
               "Try for example \"Which genomes scored highest?\", \"Did Rama rate any genomes 5?\" or "
               "\"How do I rate clips?\"")
THANKS_REPLY = "You're welcome! Ask me anything else about the genomes, the voices, the ratings or the website."
BYE_REPLY = "Goodbye! Come back any time you want to ask about the Dramatizer."

OFF_TOPIC = ("I only answer questions about the Dramatizer: its voices, genomes, listening scores, results, the website "
             "and the team. Try for example \"Which genomes work best for each voice?\" or \"How do I rate clips?\"")

# Words that make a question one about the Dramatizer (its data, results, site or team)
DOMAIN = re.compile(
    r"dramatiz|genome|\bgenes?\b|feature|sigh|laugh|pause|breath|stress|accent|speed|\bvoices?\b|speaker|dataset|"
    r"vctk|crema|libritts|\bclips?\b|\blines?\b|render|\brat(e|ed|es|ing|ings)\b|scor|\bmos\b|listen|pile|"
    r"\bplain\b|baseline|artifact|sanity|verdict|search|evaluat|result|milestone|wer\b|pesq|stoi|si-sdr|similarity|"
    r"record|evidence|chart|graph|question \d|website|\bsite\b|\bpage\b|log ?in|account|password|register|upload|"
    r"generate|studio|\btab\b|contribut|team|who made|built|cosyvoice|chatterbox|clone|cloning|tts|speech|"
    r"\bash\b|julius|\brama\b|\bmin\b|ahmed|p\d{3}\b|\b\d{4}\b", re.IGNORECASE)


def greeting(question):
    """The reply to small talk, or None when the message is a real question."""
    if not GREETING.match(question or ""):
        return None
    if THANKS.match(question):
        return THANKS_REPLY
    if BYE.match(question):
        return BYE_REPLY
    return HELLO_REPLY


def about_us(question):
    """Whether a question could be about the Dramatizer (else it is declined, not looked up)."""
    return bool(DOMAIN.search(question or ""))
