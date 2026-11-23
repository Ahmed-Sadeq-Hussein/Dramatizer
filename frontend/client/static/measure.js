// measure.js | The Lab tab: measure renders and voices side by side (POST /lab/compare): word error rate with what
// was heard word by word, speaker similarity to a reference and between every pair, and estimated quality.
"use strict";

const Measure = (() => {
  const MAX = 8;
  const M = { picked: [], reference: null, ready: false, source: "renders", lastResult: null };

  // --- the chosen recordings -------------------------------------------------------------------
  const keyOf = (s) => JSON.stringify(s);
  const audioUrl = (s) => s.kind === "render" ? `${API}/renders/${encodeURIComponent(s.id)}/audio`
    : s.kind === "voice" ? `${API}/voices/audio?dataset=${encodeURIComponent(s.dataset)}&clip_id=${encodeURIComponent(s.clip_id)}`
    : s.clip === "reference" ? `${API}/genome/searches/${encodeURIComponent(s.run)}/reference.wav`
    : `${API}/genome/searches/${encodeURIComponent(s.run)}/audio/${s.clip}.wav`;

  function add(source, label, text) {
    if (M.picked.some((p) => keyOf(p.source) === keyOf(source))) { toast("That recording is already chosen."); return; }
    if (M.picked.length >= MAX) { toast(`The Lab compares at most ${MAX} recordings at once.`, true); return; }
    M.picked.push({ source, label, text });
    if (M.reference === null && source.kind === "voice") M.reference = M.picked.length - 1;   // an original: the natural reference
    drawPicked();
  }

  function drop(i) {
    M.picked.splice(i, 1);
    if (M.reference === i) M.reference = null;
    else if (M.reference !== null && M.reference > i) M.reference -= 1;
    drawPicked();
  }

  function drawPicked() {
    const list = $("lab-picked");
    if (!M.picked.length) {
      list.replaceChildren(el("li", { class: "muted small" }, "Nothing chosen yet: add renders or original voices below."));
    } else {
      list.replaceChildren(...M.picked.map((p, i) => el("li", { class: "item lab-pick" },
        el("div", { class: "item-head" }, el("strong", {}, `${i + 1}. ${p.label}`),
          el("label", { class: "check" }, el("input", { type: "radio", name: "lab-ref", checked: M.reference === i,
            onchange: () => { M.reference = i; } }), "Reference"),
          el("button", { type: "button", class: "quiet small", onclick: () => drop(i) }, "Remove")),
        p.text ? el("p", { class: "small" }, p.text) : null,
        el("audio", { controls: true, preload: "none", src: audioUrl(p.source) }))));
    }
    $("lab-count").textContent = `${M.picked.length} of ${MAX}`;
    $("lab-run").disabled = !M.picked.length;
    $("lab-noref").checked = M.reference === null;
  }

  // --- where recordings come from: renders, original voices, search clips ---------------------
  async function findRenders() {
    const params = new URLSearchParams({ q: $("lab-rq").value.trim(), page: 1 });
    if ($("lab-rmine").checked) params.set("mine", "1");
    const box = $("lab-found");
    box.replaceChildren(el("li", { class: "muted small" }, "Loading…"));
    try {
      const { renders, total } = await call("/renders?" + params);
      if (!renders.length) { box.replaceChildren(el("li", { class: "muted small" }, "No renders match.")); return; }
      box.replaceChildren(...renders.map((r) => {
        const title = r.genome || (r.genes ? `Edited genome (${r.genes} genes)` : "Plain line");
        const label = `${title} · ${r.voice || "?"} · by ${r.by || "?"}`;
        return foundItem(label, `${r.made || ""} · ${r.text || ""}`, () => add({ kind: "render", id: r.id }, label, r.text));
      }), ...(total > renders.length ? [el("li", { class: "muted small" }, `Showing ${renders.length} of ${total}: narrow the search.`)] : []));
    } catch (error) { box.replaceChildren(el("li", { class: "muted small" }, error.message)); }
  }

  async function findVoices() {
    const params = new URLSearchParams({ dataset: $("lab-dataset").value, q: $("lab-vq").value.trim(), page: 1 });
    const box = $("lab-found");
    box.replaceChildren(el("li", { class: "muted small" }, "Loading…"));
    try {
      const { clips, total } = await call("/voices?" + params);
      if (!clips.length) { box.replaceChildren(el("li", { class: "muted small" }, "No voices match.")); return; }
      box.replaceChildren(...clips.map((c) => {
        const label = `Original · ${c.dataset}/${c.speaker}`;
        return foundItem(label, c.text, () => add({ kind: "voice", dataset: c.dataset, clip_id: c.clip_id }, label, c.text));
      }), ...(total > clips.length ? [el("li", { class: "muted small" }, `Showing ${clips.length} of ${total}: type a speaker or words.`)] : []));
    } catch (error) { box.replaceChildren(el("li", { class: "muted small" }, error.message)); }
  }

  async function findSearchClips() {
    const run = $("lab-run-id").value;
    const box = $("lab-found");
    if (!run) { box.replaceChildren(); return; }
    box.replaceChildren(el("li", { class: "muted small" }, "Loading…"));
    try {
      const { run: r } = await call(`/genome/searches/${encodeURIComponent(run)}`);
      const clip = (name, title, about) => {
        const label = `${r.voice} · ${title} · search ${run.slice(0, 20)}`;
        return foundItem(label, about, () => add({ kind: "search", run, clip: name }, label, name === "reference" ? r.reference_text : r.text));
      };
      const items = [clip("reference", "original", r.reference_text || ""), clip("baseline", "plain line", r.text)];
      for (const g of r.genomes || []) if (g.audio) items.push(clip(clipKey(g.number), `genome ${g.number}`, (g.described || []).join(" · ")));
      box.replaceChildren(...items);
    } catch (error) { box.replaceChildren(el("li", { class: "muted small" }, error.message)); }
  }

  function foundItem(label, about, onAdd) {
    return el("li", { class: "lab-found-item" },
      el("button", { type: "button", class: "quiet small", onclick: onAdd }, "Add"),
      el("span", {}, el("strong", {}, label), about ? el("span", { class: "muted small" }, " · " + about) : null));
  }

  function showSource(name) {
    M.source = name;
    document.querySelectorAll("#lab-sources button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.source === name)));
    document.querySelectorAll(".lab-source").forEach((f) => (f.hidden = f.dataset.source !== name));
    $("lab-found").replaceChildren();
    if (name === "renders") findRenders();
    if (name === "voices") findVoices();
    if (name === "searches") loadRuns();
  }

  async function loadRuns() {
    try {
      const { searches } = await call("/genome/searches");
      $("lab-run-id").replaceChildren(el("option", { value: "" }, "Choose a search…"),
        ...searches.filter((s) => !s.batch || can("rate_for_others")).map((s) =>
          el("option", { value: s.id }, `${s.voice} · ${s.created} · ${(s.text || "").slice(0, 50)}`)));
    } catch (error) { toast(error.message, true); }
  }

  // --- measuring ---------------------------------------------------------------------------------
  async function run() {
    const measure = ["wer", "similarity", "quality"].filter((m) => $("lab-m-" + m).checked);
    if (!measure.length) { toast("Tick at least one measurement.", true); return; }
    const body = { recordings: M.picked.map((p) => p.source), reference: M.reference, measure };
    const script = $("lab-script").value.trim();
    if (script) body.text = script;
    $("lab-run").disabled = true;
    $("lab-progress").textContent = "Sending…";
    try {
      const { job } = await call("/lab/compare", { body });
      follow(job.id);
    } catch (error) {
      $("lab-progress").textContent = "";
      $("lab-run").disabled = false;
      toast(error.message, true);
    }
  }

  function follow(id) {
    const tick = async () => {
      try {
        const { job } = await call(`/jobs/${id}`);
        if (job.status === "queued") $("lab-progress").textContent = queuedText(job);     // app.js
        else if (job.status === "running") $("lab-progress").textContent = "Measuring on the host… (the first time loads Whisper, the speaker encoder and SQUIM)";
        else if (["failed", "cancelled", "expired"].includes(job.status)) {
          $("lab-progress").textContent = (job.status === "failed" ? "Failed: " : "") + job.error;
          $("lab-run").disabled = false;
          return;
        }
        else if (job.status === "done") {
          $("lab-progress").textContent = `Measured in ${job.lab.seconds} s`;
          $("lab-run").disabled = false;
          show(job.lab);
          loadHistory();
          return;
        }
        setTimeout(tick, 1500);
      } catch (error) { $("lab-progress").textContent = error.message; $("lab-run").disabled = false; }
    };
    tick();
  }

  // --- the results ---------------------------------------------------------------------------------
  const num = (x, d) => (x === null || x === undefined ? "–" : Number(x).toFixed(d));

  // A value past the project's rule of thumb gets a mark and words, never colour alone
  function judged(value, text, ok, why) {
    if (value === null || value === undefined) return el("td", { class: "num" }, "–");
    return el("td", { class: "num" + (ok ? "" : " lab-off"), title: ok ? "" : why }, text, ok ? null : el("span", { class: "lab-flag" }, " ▲"));
  }

  function show(result) {
    M.lastResult = result;
    const t = result.thresholds || {};
    const rows = result.recordings;
    const head = el("tr", {}, ...["#", "Recording", "WER", "Errors", "Similarity to reference", "PESQ", "STOI", "SI-SDR (dB)",
      "Seconds", "Speech s", "Words/s"].map((h, i) => el("th", { class: i > 1 ? "num" : "" }, h)));
    const body = rows.map((r, i) => el("tr", {},
      el("td", {}, String(i + 1)),
      el("td", {}, r.label, r.reference ? el("span", { class: "tag" }, " reference") : null),
      judged(r.wer, num(r.wer, 3), r.wer === null || r.wer === undefined || r.wer <= t.wer, `Above ${t.wer}: the project's limit for an accepted clip`),
      el("td", { class: "num" }, r.errors === null || r.errors === undefined ? "–" : `${r.errors} / ${r.word_count}`),
      judged(r.similarity, num(r.similarity, 3), !(r.similarity < t.similarity), `Below ${t.similarity}: may not sound like the same person`),
      judged(r.quality && r.quality.pesq, num(r.quality && r.quality.pesq, 2), !(r.quality && r.quality.pesq < t.pesq), `Below ${t.pesq}: noisy or distorted`),
      el("td", { class: "num" }, num(r.quality && r.quality.stoi, 3)),
      el("td", { class: "num" }, num(r.quality && r.quality.si_sdr, 1)),
      el("td", { class: "num" }, num(r.seconds, 2)),
      el("td", { class: "num" }, num(r.speech_seconds, 2)),
      el("td", { class: "num" }, num(r.words_per_second, 2))));
    const parts = [
      el("p", { class: "muted small" }, `Measured ${result.made}${result.by ? " by " + result.by : ""} · ` +
        Object.entries(result.models || {}).map(([k, v]) => `${k}: ${v}`).join(" · ") +
        ` · ▲ past the rule of thumb (WER above ${t.wer}, similarity below ${t.similarity}, PESQ below ${t.pesq})`),
      el("div", { class: "table-wrap" }, el("table", { class: "lab-table" }, el("thead", {}, head), el("tbody", {}, ...body))),
    ];
    if (result.matrix) parts.push(matrix(result));
    if (rows.some((r) => r.heard !== undefined)) parts.push(el("h4", {}, "What was heard"), ...rows.map(heard));
    $("lab-result").replaceChildren(...parts);
    $("lab-result-card").hidden = false;
  }

  // Speaker similarity of every pair: one hue, darker is more alike, and the number is always written
  function matrix(result) {
    const n = result.matrix.length;
    const cell = (v, i, j) => {
      if (i === j) return el("td", { class: "num lab-diag" }, "—");
      const strength = Math.max(0, Math.min(1, (v - 0.5) / 0.5));       // 0.5 (different people) .. 1 (identical)
      return el("td", { class: "num lab-cell", style: `--s:${strength.toFixed(2)}`,
                        title: `${result.recordings[i].label} vs ${result.recordings[j].label}: ${v.toFixed(3)}` }, v.toFixed(3));
    };
    return el("div", {},
      el("h4", {}, "Speaker similarity between every pair"),
      el("p", { class: "muted small" }, `1.0 is the same recording; about ${result.thresholds.similarity}+ sounds like the same person; about 0.5 is different people.`),
      el("div", { class: "table-wrap" }, el("table", { class: "lab-table lab-matrix" },
        el("thead", {}, el("tr", {}, el("th", {}), ...result.recordings.map((_, j) => el("th", { class: "num" }, String(j + 1))))),
        el("tbody", {}, ...result.matrix.map((row, i) => el("tr", {},
          el("th", { scope: "row" }, `${i + 1}. ${result.recordings[i].label}`), ...row.map((v, j) => cell(v, i, j))))))));
  }

  function heard(r, i) {
    const words = (r.words || []).flatMap((w) => {
      if (w.op === "equal") return [el("span", {}, w.expected.join(" ") + " ")];
      if (w.op === "substitute") return [el("del", { title: "expected" }, w.expected.join(" ")), " ", el("ins", { title: "heard instead" }, w.heard.join(" ")), " "];
      if (w.op === "delete") return [el("del", { title: "missing" }, w.expected.join(" ")), " "];
      return [el("ins", { title: "extra" }, w.heard.join(" ")), " "];
    });
    return el("div", { class: "lab-heard" },
      el("p", { class: "small" }, el("strong", {}, `${i + 1}. ${r.label}`)),
      r.text ? el("p", { class: "small" }, el("span", { class: "muted" }, "Script: "), r.text) : el("p", { class: "muted small" }, "No script to compare with."),
      el("p", { class: "small" }, el("span", { class: "muted" }, "Heard: "), r.heard || "(nothing)"),
      words.length ? el("p", { class: "small lab-words" }, el("span", { class: "muted" }, "Word by word: "), ...words) : null);
  }

  async function loadHistory() {
    try {
      const { results } = await call("/lab/results");
      $("lab-history").replaceChildren(...(results.length ? results.map((r) => el("li", {},
        el("a", { href: "#", onclick: (e) => { e.preventDefault(); show(r); $("lab-result-card").scrollIntoView({ block: "start" }); } },
          `${r.made} · ${r.recordings.length} recording${r.recordings.length === 1 ? "" : "s"}`),
        el("span", { class: "muted small" }, " · " + r.recordings.map((x) => x.label).join(" | ").slice(0, 160)),
        " ", el("button", { type: "button", class: "quiet small", onclick: () => reuse(r) }, "Use these again")))
        : [el("li", { class: "muted small" }, "None yet.")]));
    } catch (error) { /* not important */ }
  }

  function reuse(result) {
    M.picked = result.recordings.filter((r) => r.source).map((r) => ({ source: r.source, label: r.label, text: r.text }));
    M.reference = result.reference;
    drawPicked();
    $("lab-picked").scrollIntoView({ block: "nearest" });
  }

  function init() {
    if (M.ready) return;
    M.ready = true;
    $("lab-sources").addEventListener("click", (e) => { const b = e.target.closest("button[data-source]"); if (b) showSource(b.dataset.source); });
    $("lab-rform").addEventListener("submit", (e) => { e.preventDefault(); findRenders(); });
    $("lab-rmine").addEventListener("change", findRenders);
    $("lab-vform").addEventListener("submit", (e) => { e.preventDefault(); findVoices(); });
    $("lab-dataset").addEventListener("change", findVoices);
    $("lab-run-id").addEventListener("change", findSearchClips);
    $("lab-noref").addEventListener("change", (e) => { if (e.target.checked) { M.reference = null; drawPicked(); } });
    $("lab-clear").addEventListener("click", () => { M.picked = []; M.reference = null; drawPicked(); });
    $("lab-run").addEventListener("click", run);
    $("lab-picked").addEventListener("play", (e) => {
      document.querySelectorAll("#tab-measure audio").forEach((a) => { if (a !== e.target) a.pause(); });
    }, true);
    document.querySelector('#lab-sources [data-source="searches"]').hidden = !can("search");
  }

  return {
    open() {
      init();
      $("lab-dataset").replaceChildren(...[...$("dataset").options].map((o) => el("option", { value: o.value }, o.textContent)));
      drawPicked();
      showSource(M.source);
      loadHistory();
    },
    add,
  };
})();
