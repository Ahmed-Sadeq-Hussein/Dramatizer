"""assistant.py | "Ask the Dramatizer" on the website: the chatbot (Gemma 4) run for the API server.

The chatbot (API calls/chatbot.py) needs a newer transformers than CosyVoice allows, so it cannot be
loaded inside the server. Instead the server starts it once, in its own Python environment, as a
worker (chatbot.py --serve), and talks to it over stdin/stdout: one question per line, one reply per line.

Every question carries the asker's own session token. The worker's tool calls go back to this API with
that token, so the model reads only what the asker may see: a listener gets their own scores, an admin
gets everything. The token never leaves this computer.

Settings (.env):
    CHATBOT_PYTHON   the Python of the LLM environment, e.g. D:/envs/llm/python.exe (unset: no model)
    CHATBOT_MODEL    a folder in LLM_LIBRARY_DIR (default gemma-4-E4B-it), or the server's model name
    CHATBOT_SERVER   an OpenAI-compatible address (Ollama, llama.cpp, vLLM) instead of a local model
    CHATBOT_PRELOAD  1 (default): load the model when the API starts; 0: on the first question

Loading Gemma takes about two minutes. Until it is ready, or
when there is no model, the server answers from the records itself (backend/rag/answer.py), with
citations, and says so; the page shows which one answered.
"""
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time

from backend import model

CHATBOT = model.PROJECT_DIR / "API calls" / "chatbot.py"
REPLY = "@@reply "
LOG_PATH = model.PROJECT_DIR / "history" / "api" / "assistant.log"
ANSWER_SECONDS = 90           # a question waits this long for the model, then the records answer (a Cloudflare
                              # tunnel drops a request after 100 s, the client app after 120 s)
START_SECONDS = 600           # loading the model may take this long


class Assistant:
    def __init__(self):
        self.url = self.fingerprint = None
        self.process = None
        self.state = "off"
        self.problem = None
        self.model = None
        self.started = None
        self._lock = threading.Lock()
        self._waiting = {}
        self._ids = itertools.count(1)

    # ------------------------------------------------------------------ settings --
    def configure(self, url, fingerprint):
        """Called by the server when it starts: where the worker reaches the API."""
        self.url, self.fingerprint = url, fingerprint

    @staticmethod
    def _optional(name):
        try:
            return model.setting(name)
        except KeyError:
            return ""

    def command(self):
        server = self._optional("CHATBOT_SERVER")
        python = self._optional("CHATBOT_PYTHON") or (sys.executable if server else "")
        if not python or not self.url:
            return None
        name = self._optional("CHATBOT_MODEL") or "gemma-4-E4B-it"
        command = [python, str(CHATBOT), "--serve", "--url", self.url, "--fingerprint", self.fingerprint, "--model", name]
        return command + (["--server", server] if server else [])

    def status(self):
        """{"state": off | starting | ready | failed, "model", "problem"}"""
        if self.state in ("starting", "ready") and self.process and self.process.poll() is not None:
            self.state, self.problem = "failed", f"the model stopped (exit {self.process.returncode}); see {LOG_PATH.name}"
        if self.state == "starting" and time.time() - (self.started or 0) > START_SECONDS:
            self.state, self.problem = "failed", "the model took too long to load"
        if self.state == "off":
            if self.command() is None:
                return {"state": "off", "model": None, "problem": "no language model is set up (CHATBOT_PYTHON in .env)"}
            return {"state": "idle", "model": None, "problem": None}      # starts on the first question
        return {"state": self.state, "model": self.model, "problem": self.problem}

    # ------------------------------------------------------------------ the worker --
    def start(self):
        """Start the worker if it is not running (returns at once; it loads in the background)."""
        with self._lock:
            if self.state in ("starting", "ready") and self.process and self.process.poll() is None:
                return
            command = self.command()
            if command is None:
                self.state = "off"
                return
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
            # The worker writes to its own copy of the log's handle; this one is closed at once, so the log is
            # not held open here (on Windows an open file cannot be deleted or moved)
            with open(LOG_PATH, "a", encoding="utf-8") as log:
                self.process = subprocess.Popen(command, cwd=str(CHATBOT.parent), stdin=subprocess.PIPE,
                                                stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8",
                                                bufsize=1, env=env,
                                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.state, self.problem, self.started = "starting", None, time.time()
            threading.Thread(target=self._read, args=(self.process,), daemon=True).start()

    def _read(self, process):
        for line in process.stdout:
            if not line.startswith(REPLY):
                continue                               # the model's loading messages
            try:
                reply = json.loads(line[len(REPLY):])
            except ValueError:
                continue
            if reply.get("ready"):
                self.state, self.model = "ready", reply.get("model")
                continue
            box = self._waiting.pop(reply.get("id"), None)
            if box:
                box.put(reply)
        if process is self.process and self.state != "off":
            self.state = "failed"
            self.problem = self.problem or f"the model stopped; see {LOG_PATH.name}"
        for box in list(self._waiting.values()):
            box.put({"error": "the model stopped"})
        self._waiting.clear()

    def ask(self, question, token, history=(), timeout=ANSWER_SECONDS):
        """The model's reply {"answer", "tools", "seconds"}, or raises RuntimeError (not ready, failed, too slow)."""
        if self.state != "ready" or not self.process or self.process.poll() is not None:
            raise RuntimeError("the language model is not ready")
        number = next(self._ids)
        box = queue.Queue()
        self._waiting[number] = box
        line = json.dumps({"id": number, "question": question, "token": token, "history": list(history)})
        with self._lock:
            self.process.stdin.write(line + "\n")
            self.process.stdin.flush()
        try:
            reply = box.get(timeout=timeout)
        except queue.Empty:
            self._waiting.pop(number, None)
            raise RuntimeError("the language model took too long") from None
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def stop(self):
        with self._lock:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)          # so its log and pipes are let go before this returns
                except subprocess.TimeoutExpired:
                    self.process.kill()
            if self.process and self.process.stdin:
                try:
                    self.process.stdin.close()             # stdout ends on its own; its reader thread closes it
                except OSError:
                    pass
            self.state = "off"


ASSISTANT = Assistant()
