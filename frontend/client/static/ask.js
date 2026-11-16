// ask.js | Home: "Ask the Dramatizer" (POST /assistant/ask) and "About the Dramatizer" (GET /about).
"use strict";

const Ask = {
  history: [],
  busy: false,
  ready: false,
  poll: null,
  examples: [
    "Which genomes work best for each voice?",
    "Which genomes can we exclude?",
    "How do I rate clips?",
    "Who made the Dramatizer?",
  ],

  init() {
    if (this.ready) return;
    this.ready = true;
    $("ask-form").addEventListener("submit", (e) => { e.preventDefault(); this.send($("ask-q").value); });
    $("ask-examples").replaceChildren(...this.examples.map((q) =>
      el("button", { type: "button", class: "quiet small", onclick: () => this.send(q) }, q)));
    this.about();
    this.status();
  },

  async about() {
    try {
      const a = await call("/about");
      $("about-what").textContent = a.what;
      $("about-how").replaceChildren(...a.how.map((s) => el("li", {}, el("strong", {}, s.step + ": "), s.text)));
      $("about-people").replaceChildren(...a.contributors.map((n) => el("li", {}, n)));
    } catch (e) {
      $("about-what").textContent = "The description could not be loaded: " + e.message;
    }
  },

  async status() {
    let s;
    try { s = await call("/assistant"); } catch (e) { return; }
    const words = {
      ready: `Answered by ${s.model || "the language model"}, which looks the facts up in the records through the API.`,
      idle: "The language model starts with your first question; until it has loaded, the records answer directly.",
      starting: "The language model is loading (about a minute). Until then, answers come straight from the records.",
      off: "No language model is set up on this host, so answers come straight from the records.",
      failed: `The language model is not running${s.problem ? " (" + s.problem + ")" : ""}; answers come from the records.`,
    };
    $("ask-status").textContent = words[s.state] || "";
    clearTimeout(this.poll);
    if (s.state === "starting") this.poll = setTimeout(() => this.status(), 5000);
  },

  bubble(who, ...content) {
    const b = el("div", { class: "ask-msg " + who }, ...content);
    $("ask-log").append(b);
    b.scrollIntoView({ block: "nearest" });
    return b;
  },

  meta(a) {
    if (a.source === "model") {
      const n = (a.tools || []).length;
      return `${a.model || "Language model"} · ${n} look-up${n === 1 ? "" : "s"}${a.seconds ? " · " + a.seconds + " s" : ""}`;
    }
    if (a.source === "about") return "From the project description";
    if (a.absent) return "From the records · nothing found";
    return `From the records · ${a.cited} record${a.cited === 1 ? "" : "s"} cited`;
  },

  async send(text) {
    const question = (text || "").trim();
    if (!question || this.busy) return;
    this.busy = true;
    $("ask-send").disabled = true;
    $("ask-q").value = "";
    this.bubble("you", question);
    const waiting = this.bubble("bot pending", "Looking it up…");
    try {
      const a = await call("/assistant/ask", { body: { question, history: this.history.slice(-6) } });
      const parts = [el("p", { class: "ask-text" }, a.answer), el("p", { class: "muted small" }, this.meta(a))];
      if (a.citations && a.citations.length) {
        parts.push(el("details", { class: "small" }, el("summary", {}, "Records used"),
          el("ul", {}, ...a.citations.map((c) => el("li", {}, el("code", {}, c.id), ` (${c.revision})`))),
          a.cited > a.citations.length ? el("p", { class: "muted" }, `and ${a.cited - a.citations.length} more`) : null));
      }
      if (a.tools && a.tools.length) {
        parts.push(el("details", { class: "small" }, el("summary", {}, "Look-ups made"),
          el("ul", {}, ...a.tools.map((t) => el("li", {}, el("code", {}, t.tool), " ", JSON.stringify(t.arguments || {}))))));
      }
      if (a.note) parts.push(el("p", { class: "muted small" }, a.note));
      waiting.replaceChildren(...parts);
      waiting.className = "ask-msg bot";
      this.history.push({ role: "user", content: question }, { role: "assistant", content: a.answer });
    } catch (e) {
      waiting.replaceChildren(el("p", {}, "No answer: " + e.message));
      waiting.className = "ask-msg bot error";
    } finally {
      this.busy = false;
      $("ask-send").disabled = false;
      $("ask-q").focus();
      this.status();
    }
  },
};
