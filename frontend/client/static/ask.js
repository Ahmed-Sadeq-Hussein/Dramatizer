// ask.js | The Ask tab: "Ask the Dramatizer" (POST /assistant/ask), with links to the genomes and voices an answer
// found, and the look-up details of each answer (what the chatbot called and what came back), and Home's About.
"use strict";

const Ask = {
  history: [],
  busy: false,
  ready: false,
  poll: null,
  examples: [
    "Which genomes work best for each voice?",
    "Did Rama rate any genomes 5?",
    "Show me genomes that performed well",
    "Which genomes can we exclude?",
    "How do I rate clips?",
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
    if (a.source === "small_talk") return "Small talk · nothing looked up";
    if (a.source === "model") {
      const n = (a.tools || []).length;
      return `${a.model || "Language model"} · ${n} look-up${n === 1 ? "" : "s"}${a.seconds ? " · " + a.seconds + " s" : ""}`;
    }
    if (a.source === "about") return "From the project description";
    if (a.source === "glossary") return `From the glossary · ${a.term}`;
    if (a.route === "off_topic") return "Not about the Dramatizer · nothing looked up";
    if (a.absent) return "From the records · nothing found";
    return `From the records · ${a.cited} record${a.cited === 1 ? "" : "s"} cited`;
  },

  // --- links: open a genome in the studio, or a voice in Voices -------------------------------
  openGenome(key) {
    Studio.show(key);
  },

  openVoice(dataset, speaker) {
    showTab("voices");
    const known = [...$("dataset").options].some((o) => o.value === dataset);
    if (!known) { toast(`${dataset} voices are not offered to your account.`, true); return; }
    $("dataset").value = dataset;
    $("voice-q").value = speaker;
    loadVoices(1);
  },

  openRender(id) {
    showTab("renders");
    $("rn-q").value = id;
    $("rn-mine").checked = false;
    Renders.open();
  },

  linkLabel(l) {
    if (l.kind === "voice") return `${l.dataset}/${l.speaker}`;
    if (l.kind === "render") return `Render · ${l.genome || "plain line"} · ${l.voice || ""}`;
    if (l.name && !l.voice) return `Genome "${l.name}" (made by hand)`;
    const bits = [`${l.voice || "?"} · genome ${l.genome}`];
    if (l.name) bits.push(`"${l.name}"`);
    if (l.score !== undefined && l.score !== null) bits.push(`score ${l.score}` + (l.listener ? ` by ${l.listener}` : ""));
    return bits.join(" · ");
  },

  linkList(links) {
    const genomes = links.filter((l) => l.kind === "genome"), voices = links.filter((l) => l.kind === "voice");
    const renders = links.filter((l) => l.kind === "render");
    const item = (l) => el("li", {}, el("a", { href: "#", onclick: (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (l.kind === "genome") this.openGenome(l.key);
      else if (l.kind === "render") this.openRender(l.id);
      else this.openVoice(l.dataset, l.speaker);
    } }, this.linkLabel(l)), l.genes && l.genes.length ? el("span", { class: "muted small" }, " · " + [].concat(l.genes).join("; ")) : null);
    const parts = [];
    if (genomes.length) parts.push(el("p", { class: "small" }, el("strong", {}, `Genomes (${genomes.length})`), " · open one in the studio"),
                                   el("ul", { class: "ask-links small" }, ...genomes.map(item)));
    if (renders.length) parts.push(el("p", { class: "small" }, el("strong", {}, `Renders (${renders.length})`), " · listen to one"),
                                   el("ul", { class: "ask-links small" }, ...renders.map(item)));
    if (voices.length) parts.push(el("p", { class: "small" }, el("strong", {}, `Voices (${voices.length})`)),
                                  el("ul", { class: "ask-links small" }, ...voices.map(item)));
    return parts;
  },

  // --- look-up details: everything the answer rests on, for checking it --------------------------
  pretty(text) {
    try { return JSON.stringify(JSON.parse(text), null, 2); } catch { return String(text || ""); }
  },

  showDebug(entry, bubble) {
    document.querySelectorAll("#ask-log .ask-msg.bot").forEach((b) => b.classList.toggle("selected", b === bubble));
    $("ask-debug-hint").hidden = true;
    const a = entry.answer;
    const row = (k, v) => (v === undefined || v === null || v === "" ? [] : [el("dt", {}, k), el("dd", {}, String(v))]);
    const parts = [el("dl", {},
      ...row("Question", entry.question), ...row("Answered by", a.source === "model" ? a.model || "language model" : a.source),
      ...row("Route", a.route), ...row("Time", a.seconds ? a.seconds + " s" : null), ...row("Note", a.note),
      ...row("Look-ups", (a.tools || []).length), ...row("Links", (a.links || []).length),
      ...row("Records cited", a.cited))];
    (a.tools || []).forEach((t, i) => parts.push(el("details", { open: i === (a.tools.length - 1) },
      el("summary", {}, `${i + 1}. `, el("code", {}, t.tool)),
      el("p", { class: "muted small" }, "Arguments"), el("pre", {}, JSON.stringify(t.arguments || {}, null, 2)),
      el("p", { class: "muted small" }, "What came back"), el("pre", {}, t.result !== undefined ? this.pretty(t.result)
                                                                         : "(not sent by this host's chatbot)"))));
    if (a.citations && a.citations.length) {
      parts.push(el("details", {}, el("summary", {}, `Records cited (${a.cited})`),
        el("ul", {}, ...a.citations.map((c) => el("li", {}, el("code", {}, c.id), ` (${c.revision}, grade ${c.grade})`)))));
    }
    if (a.links && a.links.length) parts.push(el("details", {}, el("summary", {}, "Links as data"), el("pre", {}, JSON.stringify(a.links, null, 2))));
    parts.push(el("details", {}, el("summary", {}, "The whole reply"), el("pre", {}, JSON.stringify(a, null, 2))));
    $("ask-debug").replaceChildren(...parts);
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
      const parts = [el("p", { class: "ask-text" }, a.answer), ...this.linkList(a.links || []),
                     el("p", { class: "muted small" }, this.meta(a))];
      if (a.note) parts.push(el("p", { class: "muted small" }, a.note));
      waiting.replaceChildren(...parts);
      waiting.className = "ask-msg bot";
      const entry = { question, answer: a };
      waiting.title = "Show what this answer looked up";
      waiting.addEventListener("click", () => this.showDebug(entry, waiting));
      this.showDebug(entry, waiting);
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
