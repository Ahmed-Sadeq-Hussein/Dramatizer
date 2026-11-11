// app.js | The Dramatizer client page. Every call goes to /api/v1/..., which client.py passes on to the host.
"use strict";

const API = "/api/v1";
const $ = (id) => document.getElementById(id);
const state = { me: null, voice: null, page: 1, run: null, runData: null, view: "rate", polls: {} };

// ---------------------------------------------------------------- helpers --
function el(tag, props, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (key === "class") node.className = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value !== undefined && value !== null && value !== false) node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

async function call(path, options = {}) {
  const init = { method: options.method || "GET", headers: {} };
  if (options.body !== undefined) {
    init.method = options.method || "POST";
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(options.body);
  }
  const reply = await fetch(API + path, init);
  let data = {};
  try { data = await reply.json(); } catch (e) { /* audio or empty */ }
  if (reply.status === 401 && path !== "/login" && state.mode === "login") showLogin();
  if (!reply.ok) throw new Error(data.error || `The host answered ${reply.status}.`);
  return data;
}

let toastTimer;
function toast(message, error) {
  const t = $("toast");
  t.textContent = message;
  t.className = "toast" + (error ? " error" : "");
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), error ? 7000 : 3500);
}

const pill = (verdict) => el("span", { class: "pill " + (verdict || "neutral") }, verdict ? verdict[0].toUpperCase() + verdict.slice(1) : "waiting");
const fmt = (x, d = 3) => (x === null || x === undefined ? "–" : Number(x).toFixed(d));
const clipKey = (n) => "genome_" + String(n).padStart(2, "0");
const audioOf = (run, name) => el("audio", { controls: true, preload: "none", src: `${API}/genome/searches/${encodeURIComponent(run)}/audio/${name}` });

// ------------------------------------------------------------------- tabs --
function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((s) => (s.hidden = s.id !== "tab-" + name));
  if (name === "genomes" && !state.run) loadSearches();
  if (name === "results") loadResults();
  if (name === "rate") Rate.open();
}
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ------------------------------------------------------------- connection --
const can = (permission) => !!(state.me && state.me.account && state.me.account.permissions.includes(permission));

// Returns true when someone is known to the host (an API key, or a logged-in account)
async function connect() {
  const c = $("conn");
  try {
    const local = await (await fetch("/connection")).json();
    state.mode = local.mode;
    state.host = local.host;
    $("account").hidden = local.mode !== "login" || !local.name;
    if (local.mode === "login" && !local.name) { showLogin(); return false; }
    state.me = await call("/me");
    const role = state.me.account && state.me.account.role !== "api key" ? ` (${state.me.account.role})` : "";
    c.textContent = `${state.me.name}${role} · ${local.host}` + (state.me.jobs_waiting ? ` · ${state.me.jobs_waiting} job(s) on the host` : "");
    c.className = "conn ok";
    applyPermissions();
    return true;
  } catch (error) {
    c.textContent = error.message;
    c.className = "conn bad";
    return false;
  }
}

// What a listener account cannot do is hidden rather than refused
function applyPermissions() {
  // A listener only rates: the other tabs would show other searches and distract from the listening test
  const listener = can("rate") && !can("search") && !can("speak");
  for (const tab of ["voices", "genomes", "results"]) document.querySelector(`.tabs [data-tab="${tab}"]`).hidden = listener;
  document.querySelector('.tabs [data-tab="speak"]').hidden = !can("speak");
  document.querySelector("#tab-genomes details.card").hidden = !can("search");
  document.body.classList.toggle("can-name", can("name"));
  document.body.classList.toggle("own-scores-only", !can("rate_for_others"));
}

// ------------------------------------------------------------------ login --
function showLogin() {
  state.me = null;
  $("login").hidden = false;
  $("app").hidden = true;
  $("account").hidden = true;
  $("conn").textContent = state.host ? `Not logged in · ${state.host}` : "Not logged in";
  $("conn").className = "conn";
  setTimeout(() => $("login-user").focus(), 0);
}

$("login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const button = $("login-form").querySelector("button");
  button.disabled = true;
  $("login-error").hidden = true;
  try {
    await call("/login", { body: { username: $("login-user").value.trim(), password: $("login-pass").value } });
    $("login-pass").value = "";
    $("login").hidden = true;
    $("app").hidden = false;
    if (await connect()) start();
  } catch (error) {
    $("login-error").textContent = error.message;
    $("login-error").hidden = false;
  } finally { button.disabled = false; }
});

$("logout").addEventListener("click", async () => {
  try { await call("/logout", { body: {} }); } catch (e) {}
  location.reload();
});

$("password-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const current = $("pw-current").value, next = $("pw-new").value;
  if (next !== $("pw-again").value) return toast("The two new passwords are not the same.", true);
  try {
    const reply = await call("/me/password", { body: { current, new: next } });
    toast(reply.message);
    setTimeout(() => location.reload(), 1500);
  } catch (error) { toast(error.message, true); }
});

// -------------------------------------------------------------- 1. voices --
async function loadDatasets() {
  try {
    const { datasets } = await call("/datasets");
    $("dataset").replaceChildren(...datasets.map((d) => el("option", { value: d.name }, `${d.name} (${d.speakers} speakers)`)));
    loadVoices(1);
  } catch (error) { toast(error.message, true); }
}

async function loadVoices(page) {
  const params = new URLSearchParams({ dataset: $("dataset").value, q: $("voice-q").value.trim(), page });
  try {
    const data = await call("/voices?" + params);
    state.page = data.page;
    $("voice-count").textContent = `${data.total} clips`;
    $("page").textContent = `Page ${data.page} of ${data.pages}`;
    $("prev").disabled = data.page <= 1;
    $("next").disabled = data.page >= data.pages;
    $("voice-list").replaceChildren(...data.clips.map(voiceItem));
  } catch (error) { toast(error.message, true); }
}

function voiceItem(clip) {
  const details = Object.values(clip.details).filter(Boolean).join(" · ");
  const src = `${API}/voices/audio?` + new URLSearchParams({ dataset: clip.dataset, clip_id: clip.clip_id });
  return el("li", { class: "item" },
    el("div", { class: "item-head" },
      el("strong", {}, `${clip.speaker} · ${clip.dataset}`, details ? el("span", { class: "muted small" }, " · " + details) : null),
      el("button", { type: "button", onclick: () => chooseVoice(clip, src) }, "Use this voice")),
    el("p", {}, clip.text),
    el("audio", { controls: true, preload: "none", src }));
}

function chooseVoice(clip, src) {
  state.voice = clip;
  $("chosen").hidden = false;
  $("chosen-name").textContent = `${clip.speaker} · ${clip.dataset}`;
  $("chosen-text").textContent = clip.text;
  $("chosen-audio").src = src;
  toast(`Voice chosen: ${clip.speaker}. Now Speak or start a genome search.`);
}

$("voice-search").addEventListener("submit", (e) => { e.preventDefault(); loadVoices(1); });
$("dataset").addEventListener("change", () => loadVoices(1));
$("prev").addEventListener("click", () => loadVoices(state.page - 1));
$("next").addEventListener("click", () => loadVoices(state.page + 1));

function needVoice() {
  if (state.voice) return true;
  toast("Choose a voice first (tab 1).", true);
  showTab("voices");
  return false;
}

// --------------------------------------------------------------- 2. speak --
$("speak-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!needVoice()) return;
  const body = { dataset: state.voice.dataset, clip_id: state.voice.clip_id, text: $("speak-text").value };
  if ($("speak-seed").value !== "") body.seed = Number($("speak-seed").value);
  try {
    const { job } = await call("/speak", { body });
    const card = el("div", { class: "item" },
      el("div", { class: "item-head" }, el("strong", {}, state.voice.speaker), el("span", { class: "progress" }, "Queued…")),
      el("p", {}, body.text));
    $("speak-jobs").prepend(card);
    followJob(job.id, card);
  } catch (error) { toast(error.message, true); }
});

function followJob(id, card) {
  const progress = card.querySelector(".progress");
  const tick = async () => {
    try {
      const { job } = await call(`/jobs/${id}`);
      if (job.status === "queued") progress.textContent = `Waiting: number ${job.place_in_queue} in the queue`;
      else if (job.status === "running") progress.textContent = "Speaking on the host… (the first one loads the model, about a minute)";
      else if (job.status === "failed") { progress.textContent = "Failed: " + job.error; return; }
      else if (job.status === "done") {
        progress.textContent = `Done in ${job.seconds} s`;
        card.append(el("audio", { controls: true, src: job.audio }),
                    el("a", { href: job.audio, download: "" }, "Download"));
        return;
      }
      setTimeout(tick, 2000);
    } catch (error) { progress.textContent = error.message; }
  };
  tick();
}

// ------------------------------------------------------------- 3. genomes --
async function loadSearches() {
  try {
    const chosen = $("pile-filter").value;   // "batch|pile" or "" for all
    const [batch, pile] = chosen ? chosen.split("|") : [];
    const query = chosen ? `?batch=${encodeURIComponent(batch)}&pile=${encodeURIComponent(pile)}` : "";
    const [{ searches, piles }, { named }] = await Promise.all([call("/genome/searches" + query), call("/genome/named")]);
    const options = [el("option", { value: "" }, "All searches")];
    for (const [b, ps] of Object.entries(piles || {})) for (const p of ps)
      options.push(el("option", { value: `${b}|${p}` }, `${b} · pile ${p}`));
    $("pile-filter").replaceChildren(...options);
    $("pile-filter").value = options.some((o) => o.value === chosen) ? chosen : "";
    $("pile-note").hidden = !chosen;
    try { localStorage.setItem("pile", $("pile-filter").value); } catch (e) {}
    $("search-list").replaceChildren(...(searches.length ? searches.map((s) =>
      el("li", { class: "item clickable", onclick: () => openRun(s.id) },
        el("div", { class: "item-head" },
          el("strong", {}, s.pile ? `${s.voice} · pile ${s.pile}` : `${s.voice} · ${s.created}`),
          el("span", {},
            s.rated ? el("span", { class: "mos" }, "rated") : null, " ",
            el("span", { class: "pill " + (s.summary.done === s.summary.total ? "fine" : "neutral") },
               `${s.summary.fine} fine of ${s.summary.total}` + (s.status === "finished" ? "" : ` · ${s.status}`)))),
        el("p", { class: "muted small" }, s.text))) : [el("li", { class: "muted" }, "No searches yet.")]));
    $("named-list").replaceChildren(...(named.length ? named.slice().reverse().map((n) =>
      el("li", { class: "item clickable", onclick: () => openRun(n.run, "ranked") },
        el("div", { class: "item-head" }, el("strong", {}, n.name),
          n.mos ? el("span", { class: "mos" }, "MOS " + fmt(n.mos.mos, 2)) : null),
        el("p", { class: "muted small" }, `Genome ${n.genome} of ${n.voice}` + (n.named_by ? ` · named by ${n.named_by}` : "")),
        el("p", { class: "markup" }, n.markup))) : [el("li", { class: "muted" }, "None yet: rate a search, then pick its best genome.")]));
  } catch (error) { toast(error.message, true); }
}

$("pile-filter").addEventListener("change", loadSearches);
try { const saved = localStorage.getItem("pile"); if (saved) $("pile-filter").append(el("option", { value: saved }, saved)), $("pile-filter").value = saved; } catch (e) {}

$("search-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!needVoice()) return;
  const body = { dataset: state.voice.dataset, clip_id: state.voice.clip_id, text: $("search-text").value,
                 genes: Number($("search-genes").value) };
  if ($("search-seed").value !== "") body.seed = Number($("search-seed").value);
  try {
    const { run } = await call("/genome/searches", { body });
    toast("Search queued. The clips appear here as they are made.");
    openRun(run);
  } catch (error) { toast(error.message, true); }
});

$("back").addEventListener("click", () => {
  state.run = null;
  clearTimeout(state.polls.run);
  $("run-view").hidden = true;
  $("genome-home").hidden = false;
  loadSearches();
});

document.querySelectorAll(".views button").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));

function setView(view) {
  state.view = view;
  document.querySelectorAll(".views button").forEach((b) => b.classList.toggle("on", b.dataset.view === view));
  $("rate-form").hidden = view !== "rate";
  $("ranked").hidden = view !== "ranked";
}

async function openRun(id, view) {
  state.run = id;
  showTab("genomes");
  $("genome-home").hidden = true;
  $("run-view").hidden = false;
  if (view) setView(view);
  await refreshRun(true);
}

async function refreshRun(first) {
  clearTimeout(state.polls.run);
  const id = state.run;
  if (!id) return;
  let data;
  try { data = await call(`/genome/searches/${encodeURIComponent(id)}`); }
  catch (error) { toast(error.message, true); return; }
  if (state.run !== id) return;
  state.runData = data;
  const { run, summary } = data;
  const busy = data.state === "queued or running";
  $("run-title").textContent = `${run.voice} · ${run.genes_per_genome} genes · genome seed ${run.seed}`;
  $("run-line").textContent = run.text;
  $("run-state").textContent = busy ? `Working on the host: ${summary.done} of ${summary.total} judged… (you can rate when it has finished)`
    : data.state === "finished" ? `Finished: ${summary.fine} fine, ${summary.terrible} terrible` + (summary.failed ? `, ${summary.failed} failed` : "")
    : `Stopped after ${summary.done} of ${summary.total}.`;
  if (!busy && data.state !== "finished") {
    $("run-state").append(" ", el("button", { type: "button", class: "quiet", onclick: continueRun }, "Continue"));
  }
  const listening = [...document.querySelectorAll("#run-view audio")].some((a) => !a.paused);
  const editing = document.activeElement && document.activeElement.closest && document.activeElement.closest("#rate-form, #ranked");
  if (first || !(listening || editing)) {
    renderRate(data, busy);
    renderRanked(data);
  }
  if (busy) state.polls.run = setTimeout(() => refreshRun(false), 5000);
}

async function continueRun() {
  try { await call(`/genome/searches/${encodeURIComponent(state.run)}/continue`, { body: {} }); refreshRun(true); }
  catch (error) { toast(error.message, true); }
}

// Rating: up to 4 listeners, a slider each per clip
function raterNames(data) {
  const saved = data.ratings.raters;
  const typed = [...document.querySelectorAll("#rater-names input")].map((i) => i.value.trim());
  const names = [];
  if (!can("rate_for_others") && state.me) return [state.me.name];    // an account rates only as itself
  for (let i = 0; i < data.max_raters; i++) names.push(typed[i] || saved[i] || (i === 0 && state.me ? state.me.name : ""));
  return names;
}

function renderRate(data, busy) {
  const names = raterNames(data);
  const count = names.length === 1 ? 1 : Math.max(1, Math.min(data.max_raters, Number($("rater-count").value) || data.ratings.raters.length || 1));
  $("rater-count").value = String(count);
  $("rater-names").replaceChildren(...names.map((name, i) =>
    el("label", { dataset: { rater: i + 1 }, hidden: i >= count }, `Listener ${i + 1}`,
      el("input", { type: "text", maxlength: 40, value: name, placeholder: "Name", readonly: !can("rate_for_others") && state.me, oninput: () => relabel() }))));

  const { run } = data;
  const clips = [];
  if (run.baseline.audio) clips.push({ key: "baseline", title: "Baseline", about: "the line with no genes", verdict: null, audio: "baseline.wav", base: true });
  for (const g of run.genomes) if (g.audio) clips.push({ key: clipKey(g.number), title: `Genome ${g.number}`, about: g.described.join(" · "),
                                                          markup: g.markup + (g.speed !== 1 ? ` · speed ${g.speed}` : ""), verdict: g.verdict, audio: g.audio });
  $("rate-clips").replaceChildren(...clips.map((c) => {
    const sliders = names.map((name, i) => {
      const given = name ? (data.ratings.scores[name] || {})[c.key] : undefined;
      const out = el("output", {}, given === undefined ? "–" : String(given));
      const range = el("input", { type: "range", min: 0, max: 5, step: 0.5, value: given === undefined ? 2.5 : given,
                                  "aria-label": `Score of listener ${i + 1}, 0 to 5` });
      const box = el("label", { class: "slider" + (given === undefined ? "" : " counted"), dataset: { rater: i + 1, clip: c.key }, hidden: i >= count },
        el("span", {}, name || `Listener ${i + 1}`), range, out);
      range.addEventListener("input", () => { box.classList.add("counted"); out.textContent = String(Number(range.value)); });
      return box;
    });
    return el("li", { class: "clip " + (c.base ? "baseline" : c.verdict || "") },
      el("div", { class: "item-head" }, el("strong", {}, c.title), c.base ? el("span", { class: "muted small" }, c.about) : pill(c.verdict)),
      c.base ? null : el("p", {}, c.about),
      c.markup ? el("p", { class: "markup" }, c.markup) : null,
      audioOf(run.id, c.audio),
      el("div", { class: "sliders" }, sliders));
  }));
  $("rate-form").querySelector("button[type=submit]").disabled = busy || !clips.length;
}

function relabel() {
  document.querySelectorAll("#rater-names label").forEach((label) => {
    const name = label.querySelector("input").value.trim() || `Listener ${label.dataset.rater}`;
    document.querySelectorAll(`.slider[data-rater="${label.dataset.rater}"] span`).forEach((s) => (s.textContent = name));
  });
}

$("rater-count").addEventListener("change", () => {
  const n = Number($("rater-count").value);
  document.querySelectorAll("#run-view [data-rater]").forEach((node) => (node.hidden = Number(node.dataset.rater) > n));
});

$("rate-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const n = Number($("rater-count").value);
  const scores = {};
  const missing = [];
  for (let i = 1; i <= n; i++) {
    const name = document.querySelector(`#rater-names label[data-rater="${i}"] input`).value.trim();
    const given = {};
    document.querySelectorAll(`.slider.counted[data-rater="${i}"]`).forEach((s) => (given[s.dataset.clip] = Number(s.querySelector("input").value)));
    if (!Object.keys(given).length) continue;
    if (!name) { missing.push(i); continue; }
    scores[name] = Object.assign(scores[name] || {}, given);
  }
  if (missing.length) return toast(`Give listener ${missing.join(" and ")} a name.`, true);
  if (!Object.keys(scores).length) return toast("Move a slider for at least one clip.", true);
  try {
    const reply = await call(`/genome/searches/${encodeURIComponent(state.run)}/ratings`, { body: { scores } });
    toast(`Saved ${reply.saved} scores.`);
    await refreshRun(true);
    setView("ranked");
  } catch (error) { toast(error.message, true); }
});

// Ranked by MOS, and picking a name
function renderRanked(data) {
  const r = data.ranking;
  $("ranked-head").textContent = `Mean opinion score (0 to 5) of ${r.raters.length} listener(s)` +
    (r.raters.length ? `: ${r.raters.join(", ")}` : "") + ". Baseline: " + (r.baseline ? fmt(r.baseline.mos, 2) : "not rated") +
    (r.rated ? "." : ". No genome is rated yet: rate them under Rate.");
  $("ranked-clips").replaceChildren(...r.genomes.map((g) => {
    const name = el("input", { type: "text", maxlength: 60, placeholder: "Name this genome", value: g.picked || "", "aria-label": `Name of genome ${g.number}` });
    const pickButton = el("button", { type: "button", onclick: () => pick(g.number, name.value) }, g.picked ? "Rename" : "Pick and name");
    return el("li", { class: "clip " + (g.verdict || "") + (g.picked ? " picked" : "") },
      el("div", { class: "item-head" },
        el("strong", {}, (g.rank ? `#${g.rank} · ` : "") + `Genome ${g.number} `, g.picked ? el("span", { class: "picked-name" }, `“${g.picked}”`) : null),
        el("span", {}, g.mos ? el("span", { class: "mos" }, "MOS " + fmt(g.mos.mos, 2)) : el("span", { class: "muted small" }, "not rated"), " ", pill(g.verdict))),
      g.mos ? el("p", { class: "muted small" }, Object.entries(g.mos.scores).map(([k, v]) => `${k} ${v}`).join(" · ") +
        (g.above_baseline !== null ? ` · ${g.above_baseline >= 0 ? "+" : ""}${fmt(g.above_baseline, 2)} against the baseline` : "")) : null,
      el("p", {}, g.described.join(" · ")),
      el("p", { class: "markup" }, g.markup + (g.speed !== 1 ? ` · speed ${g.speed}` : "")),
      g.audio ? audioOf(data.run.id, g.audio) : null,
      el("div", { class: "pick" }, name, pickButton,
        g.picked ? el("button", { type: "button", class: "quiet", onclick: () => unpick(g.number) }, "Remove name") : null));
  }));
}

async function pick(number, name) {
  if (!name.trim()) return toast("Type a name first.", true);
  try {
    const { named } = await call(`/genome/searches/${encodeURIComponent(state.run)}/pick`, { body: { genome: number, name } });
    toast(`Genome ${named.genome} is now “${named.name}”.`);
    refreshRun(true);
  } catch (error) { toast(error.message, true); }
}

async function unpick(number) {
  try { await call(`/genome/searches/${encodeURIComponent(state.run)}/unpick`, { body: { genome: number } }); refreshRun(true); }
  catch (error) { toast(error.message, true); }
}

// --------------------------------------------------------------- results --
async function loadResults() {
  const box = $("results");
  box.replaceChildren(el("p", { class: "muted" }, "Loading…"));
  try {
    const { baselines, milestone2 } = await call("/results");
    const parts = [];
    for (const b of baselines.slice().reverse()) {
      const groups = Object.entries(b.summary).filter(([name]) => name !== "by_line");
      const passed = b.checks.filter((c) => c.passed).length;
      parts.push(el("h3", {}, `Baseline ${b.name}`),
        el("p", { class: "muted small" }, `${b.voices} voices × ${(b.lines || []).length} lines · made ${b.made || "?"} · ` +
          `checks ${passed} of ${b.checks.length} passed` + (b.git_dirty ? " · ran on uncommitted code" : "")),
        el("div", { class: "table-wrap" }, el("table", {},
          el("thead", {}, el("tr", {}, ["", "WER", "Heard exactly", "Similarity (95 % interval)", "PESQ"].map((h) => el("th", {}, h)))),
          el("tbody", {}, groups.map(([name, s]) => el("tr", {},
            el("td", {}, name), el("td", {}, fmt(s.wer && s.wer.mean)),
            el("td", {}, Math.round(100 * s.wer_zero) + " %"),
            el("td", {}, s.similarity ? `${fmt(s.similarity.mean)} (${fmt(s.similarity.low)}–${fmt(s.similarity.high)})` : "–"),
            el("td", {}, fmt(s.pesq && s.pesq.mean, 2))))))));
    }
    for (const m of milestone2) {
      const s = m.summary;
      parts.push(el("h3", {}, `Genome search ${m.name}`),
        el("p", {}, `${Math.round(100 * s.shares.all.fine)} % of ${s.shares.all.judged} genomes judged fine. ` +
          `Works as a guard: ${s.verdicts.works_as_a_guard ? "yes" : "no"}. Beats the baseline: decided by blind listening.`),
        el("div", { class: "table-wrap" }, el("table", {},
          el("thead", {}, el("tr", {}, ["Fine genomes minus baseline", "Mean", "95 % interval", "p"].map((h) => el("th", {}, h)))),
          el("tbody", {}, Object.entries(s.fine_minus_baseline).filter(([, d]) => d).map(([metric, d]) => el("tr", {},
            el("td", {}, metric), el("td", {}, fmt(d.mean)), el("td", {}, `${fmt(d.low)} to ${fmt(d.high)}`),
            el("td", {}, d.wilcoxon_p === null ? "–" : Number(d.wilcoxon_p).toPrecision(2))))))));
    }
    box.replaceChildren(...(parts.length ? parts : [el("p", { class: "muted" }, "No results on the host yet.")]));
  } catch (error) { box.replaceChildren(el("p", { class: "muted" }, error.message)); }
}

// ------------------------------------------------------------------ start --
let started = false;
function start() {
  if (started) {
    // Logged in again (the session ran out): reload the pile, which also sends any scores still waiting
    if (!$("tab-rate").hidden) Rate.open();
    return;
  }
  started = true;
  loadDatasets();
  // Whoever has a pile to rate (or may do nothing else) starts on the Rate genomes tab, on their pile
  const account = state.me && state.me.account;
  if (account && account.permissions.includes("rate") &&
      ((account.assignments || []).length || (!can("speak") && !can("search")))) showTab("rate");
}
connect().then((ok) => ok && start());
setInterval(() => { if (state.me) connect(); }, 30000);
