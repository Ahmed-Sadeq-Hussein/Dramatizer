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
// Copy text to the clipboard (the page is on https or 127.0.0.1, where the browser allows it; else the old way)
async function copyText(text, what) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const box = el("textarea", { style: "position: fixed; opacity: 0" });
    box.value = text;
    document.body.append(box);
    box.select();
    document.execCommand("copy");
    box.remove();
  }
  toast(`${what} copied.`);
}

// A genome copied from the site: {"dramatizer_genome": 1, "genes": [...], "from": "..."}; a bare list of genes works too
function genomeText(genes, from) {
  return JSON.stringify({ dramatizer_genome: 1, from, genes }, null, 1);
}

function readGenome(text) {
  let data;
  try { data = JSON.parse(String(text || "").trim()); } catch { throw new Error("That is not a copied genome (it is not JSON)."); }
  const genes = Array.isArray(data) ? data : data && data.genes;
  if (!Array.isArray(genes) || !genes.every((g) => g && typeof g.feature === "string")) {
    throw new Error("That is not a copied genome: it needs a list of genes, each with a feature.");
  }
  return { genes: genes.map((g) => ({ ...g })), from: (data && data.from) || null };
}

// One gene in words: "sigh at 32% of the line", "pause (realization) at 19%", "speed × 1.1"
function geneWords(g) {
  const name = String(g.feature).replace(/_/g, " ");
  if (g.feature === "speed") return `speed × ${g.rate}`;
  const where = g.position === undefined ? "" : ` at ${Math.round(Number(g.position) * 100)}% of the line`;
  return `${name}${g.kind ? ` (${g.kind})` : ""}${where}`;
}

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

// ------------------------------------------------------------------ modes --
// A mode is a set of tabs for one kind of work; a tab can be in more than one mode.
const MODES = {
  home: { tabs: ["home"], note: "" },
  rate: { tabs: ["rate"], note: "Choose a pile of clips meant for rating, and score each one.", needs: "rate" },
  generate: { tabs: ["voices", "studio"], note: "Choose a voice, then optionally a genome, and render a line." },
  listen: { tabs: ["voices", "renders"], note: "Listen to a voice in the data, or to the renders people have made." },
  ask: { tabs: ["ask"], note: "Ask about genomes, voices, ratings and results; see what each answer looked up." },
  measure: { tabs: ["measure"], note: "Measure renders and voices side by side: word error rate, speaker similarity and quality." },
  research: { tabs: ["genomes", "speak", "results", "graphs"], note: "Genome searches, their rankings, the measured results and the graphs.",
              needs: "search" },
};
const MODE_KEY = "dramatizer-mode";
let mode = null;
const lastTab = {};          // each mode reopens on the tab last used in it

const modeTabs = (name) => MODES[name].tabs;
const usableModes = () => Object.keys(MODES).filter((m) => !MODES[m].needs || can(MODES[m].needs));

function applyMode(name) {
  mode = name;
  try { localStorage.setItem(MODE_KEY, name); } catch { /* private window: not remembered */ }
  document.body.dataset.mode = name;
  document.querySelectorAll(".modes button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === name)));
  $("mode-note").textContent = MODES[name].note;
  const tabs = modeTabs(name);
  document.querySelectorAll(".tabs button").forEach((b) => (b.hidden = !tabs.includes(b.dataset.tab)));
  document.querySelector(".tabs").hidden = tabs.length < 2;      // one page: no tab bar
}

function setMode(name) {
  if (!modeTabs(name).length) return;
  applyMode(name);
  const tabs = modeTabs(name);
  const current = document.querySelector(".tabs button.on");
  showTab(tabs.includes(lastTab[name]) ? lastTab[name] : current && !current.hidden ? current.dataset.tab : tabs[0]);
}

// Show the modes this account may use. The page opens on Home; later checks keep the current page as it is.
function renderModes() {
  const usable = usableModes();
  document.querySelectorAll(".modes button").forEach((b) => (b.hidden = !usable.includes(b.dataset.mode)));
  $("home-research").hidden = !usable.includes("research");
  document.querySelector('.home-card[data-go="rate"]').hidden = !usable.includes("rate");
  if (!mode || !usable.includes(mode)) setMode("home");
  else applyMode(mode);
}
document.querySelectorAll(".home-card").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.go)));
document.querySelectorAll(".modes button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));

// ------------------------------------------------------------------- tabs --
function showTab(name) {
  // A tab outside the current mode (e.g. a listener sent to Rate genomes): switch to a mode that has it
  if (mode && !modeTabs(mode).includes(name)) {
    const other = usableModes().find((m) => modeTabs(m).includes(name));
    if (other) applyMode(other);
  }
  if (mode) lastTab[mode] = name;
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((s) => (s.hidden = s.id !== "tab-" + name));
  if (name === "genomes" && !state.run) loadSearches();
  if (name === "results") loadResults();
  if (name === "graphs") loadGraphs();
  if (name === "renders") Renders.open();
  if (name === "rate") Rate.open();
  if (name === "studio") Studio.open();
  if (name === "measure" && typeof Measure !== "undefined") Measure.open();
  if (name === "ask" && typeof Ask !== "undefined") { Ask.init(); setTimeout(() => $("ask-q").focus(), 0); }
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
    if (state.me.agreement_needed) {
      c.textContent = `${state.me.name} · ${local.host}`;
      c.className = "conn";
      askAgreement();
      return false;
    }
    const role = state.me.account && state.me.account.role !== "api key" ? ` (${state.me.account.role})` : "";
    c.textContent = `${state.me.name}${role} · ${local.host}` + (state.me.jobs_waiting ? ` · ${state.me.jobs_waiting} job(s) on the host` : "");
    c.className = "conn ok";
    applyPermissions();
    loadExtras();
    return true;
  } catch (error) {
    c.textContent = error.message;
    c.className = "conn bad";
    return false;
  }
}

// Every login sees every mode and tab. What an account may not do is disabled with a note
// (data-needs / data-lacks), and the host refuses it anyway.
function applyPermissions() {
  document.querySelectorAll("[data-needs]").forEach((b) => (b.disabled = !can(b.dataset.needs)));
  document.querySelectorAll("[data-lacks]").forEach((n) => (n.hidden = can(n.dataset.lacks)));
  document.querySelector("#tab-genomes details.card").hidden = !can("search");
  document.body.classList.toggle("can-name", can("name"));
  document.body.classList.toggle("own-scores-only", !can("rate_for_others"));
  renderModes();
}

// Extensions the host offers this account only (/me "extras"), each a script of its own, loaded once
const loadedExtras = new Set();
function loadExtras() {
  for (const name of (state.me && state.me.extras) || []) {
    if (loadedExtras.has(name) || !/^[a-z]+$/.test(name)) continue;
    loadedExtras.add(name);
    document.body.append(el("script", { src: `${API}/${name}/${name}.js` }));
  }
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
  call("/register").then((r) => { $("register-offer").hidden = !r.open; }).catch(() => {});
}

// ------------------------------------------------------------- agreement --
// The user agreement (GET /agreement): accepted when registering, and at login by accounts that have not
// accepted its current version (the host refuses them everything else until they do)
let agreement = null;
async function showAgreement(box) {
  try {
    agreement = agreement || await call("/agreement");
    box.replaceChildren(el("h3", {}, `${agreement.title} · version ${agreement.version}`),
      ...agreement.sections.flatMap((s) => [el("h4", {}, s.heading), el("p", {}, s.text)]));
  } catch (error) { box.replaceChildren(el("p", { class: "muted" }, error.message)); }
}

function askAgreement() {
  $("login").hidden = true;
  $("app").hidden = true;
  $("agree").hidden = false;
  $("agree-box").checked = false;
  showAgreement($("agree-text"));
}

$("agree-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const button = $("agree-form").querySelector("button[type=submit]");
  button.disabled = true;
  $("agree-error").hidden = true;
  try {
    await call("/me/agreement", { body: { version: agreement && agreement.version } });
    $("agree").hidden = true;
    $("app").hidden = false;
    if (await connect()) start();
  } catch (error) {
    $("agree-error").textContent = error.message;
    $("agree-error").hidden = false;
  } finally { button.disabled = false; }
});
$("agree-logout").addEventListener("click", async () => {
  try { await call("/logout", { body: {} }); } catch (e) {}
  location.reload();
});

// ------------------------------------------------------------------ register --
function showForm(register) {
  $("login-form").hidden = register;
  $("register-form").hidden = !register;
  if (register) showAgreement($("reg-agreement"));
  setTimeout(() => $(register ? "reg-user" : "login-user").focus(), 0);
}
$("show-register").addEventListener("click", () => showForm(true));
$("show-login").addEventListener("click", () => showForm(false));

$("register-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const button = $("register-form").querySelector("button[type=submit]");
  $("register-error").hidden = true;
  $("register-done").hidden = true;
  if ($("reg-pass").value !== $("reg-again").value) {
    $("register-error").textContent = "The two passwords are not the same.";
    $("register-error").hidden = false;
    return;
  }
  button.disabled = true;
  try {
    const reply = await call("/register", { body: {
      username: $("reg-user").value.trim().toLowerCase(), display_name: $("reg-name").value.trim(),
      email: $("reg-email").value.trim(), password: $("reg-pass").value,
      agreement: $("reg-agree").checked && agreement ? agreement.version : null } });
    $("reg-pass").value = $("reg-again").value = "";
    $("login-user").value = reply.username;
    $("register-done").textContent = reply.message;
    $("register-done").hidden = false;
  } catch (error) {
    $("register-error").textContent = error.message;
    $("register-error").hidden = false;
  } finally { button.disabled = false; }
});

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
    const { datasets, custom } = await call("/datasets");
    // Custom voices: people's own recordings, offered only to the voice's owner and to admins (the API decides)
    $("dataset").replaceChildren(...datasets.map((d) => el("option", { value: d.name }, `${d.name} (${d.speakers} speakers)`)),
      ...(custom ? [el("option", { value: "Custom voices" }, `Custom voices (${custom} ${custom === 1 ? "voice" : "voices"}, private)`)] : []));
    loadVoices(1);
  } catch (error) { toast(error.message, true); }
  loadMyVoices();
}

// Your uploaded voices (POST /me/voices): only this account (and admins) can use them, under Custom voices
async function loadMyVoices() {
  if (!state.me || !state.me.account || !state.me.account.username) { $("my-voices").hidden = true; return; }
  try {
    const data = await call("/me/voices");
    $("my-voices").hidden = !data.available;
    const l = data.limits;
    $("up-limits").textContent = `${l.formats.map((f) => f.slice(1)).join(", ")} · up to ${l.max_mb} MB · `
      + `${l.seconds[0]} to ${l.seconds[1]} seconds · at most ${l.per_account} voices (you have ${data.voices.length})`;
    $("my-voice-list").replaceChildren(...data.voices.map((v) => el("li", { class: "item" },
      el("div", { class: "item-head" },
        el("strong", {}, v.name, el("span", { class: "muted small" }, ` · ID ${v.speaker} · ${v.seconds} s · uploaded ${v.uploaded}`)),
        el("span", { class: "row" },
          el("button", { type: "button", onclick: () => useMyVoice(v) }, "Use this voice"),
          el("button", { type: "button", class: "quiet", onclick: () => renameMyVoice(v) }, "Rename"),
          el("button", { type: "button", class: "quiet danger", onclick: () => removeMyVoice(v) }, "Delete"))))));
  } catch (error) { $("my-voice-list").replaceChildren(el("li", { class: "muted" }, error.message)); }
}

async function useMyVoice(v) {
  $("dataset").value = "Custom voices";
  $("voice-q").value = v.speaker;
  await loadVoices(1);
  const data = await call("/voices?" + new URLSearchParams({ dataset: "Custom voices", q: v.speaker }));
  const clip = data.clips.find((c) => c.speaker === v.speaker);
  if (clip) chooseVoice(clip, `${API}/voices/audio?` + new URLSearchParams({ dataset: clip.dataset, clip_id: clip.clip_id }));
}

async function renameMyVoice(v) {
  const name = window.prompt(`A new name for "${v.name}".\nIts ID stays "${v.speaker}", so your renders with it keep working.`, v.name);
  if (name === null || name.trim() === v.name) return;
  try {
    const { voice } = await call("/me/voices/rename", { body: { folder: v.folder, name } });
    toast(`Renamed to ${voice.name}.`);
    if (state.voice && state.voice.speaker === v.speaker) state.voice.details.name = voice.name;
    loadDatasets();
  } catch (error) { toast(error.message, true); }
}

async function removeMyVoice(v) {
  if (!window.confirm(`Delete your voice "${v.name}" from the host?\n\nIt cannot be brought back. Renders you made with it stay in your renders.`)) return;
  try {
    await call("/me/voices/remove", { body: { folder: v.folder } });
    toast(`Deleted ${v.name}.`);
    if (state.voice && state.voice.speaker === v.speaker) { state.voice = null; $("chosen").hidden = true; }
    loadDatasets();
  } catch (error) { toast(error.message, true); }
}

const asBase64 = (file) => new Promise((resolve, reject) => {
  const reader = new FileReader();
  reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
  reader.onerror = () => reject(new Error("The file could not be read."));
  reader.readAsDataURL(file);
});

$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = $("up-file").files[0];
  if (!file) return;
  if (file.size > 8 * 1024 * 1024) return toast("The file is larger than 8 MB.", true);
  const button = $("upload-form").querySelector("button[type=submit]");
  button.disabled = true;
  try {
    const { voice } = await call("/me/voices", { body: { name: $("up-name").value.trim(), script: $("up-script").value.trim(),
      filename: file.name, audio: await asBase64(file), consent: $("up-consent").checked } });
    $("upload-form").reset();
    toast(`Uploaded ${voice.name}: it is under Dataset: Custom voices, for you only.`);
    loadDatasets();
  } catch (error) { toast(error.message, true); }
  finally { button.disabled = !can("speak"); }
});

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

// An uploaded voice shows the name its owner gave it, and its fixed ID with the details
const voiceName = (clip) => (clip.details && clip.details.name) || clip.speaker;

function voiceItem(clip) {
  const details = [clip.details.name && clip.details.name !== clip.speaker ? `ID ${clip.speaker}` : null,
                   ...Object.entries(clip.details).filter(([k]) => k !== "name").map(([, v]) => v)].filter(Boolean).join(" · ");
  const src = `${API}/voices/audio?` + new URLSearchParams({ dataset: clip.dataset, clip_id: clip.clip_id });
  return el("li", { class: "item" },
    el("div", { class: "item-head" },
      el("strong", {}, `${voiceName(clip)} · ${clip.dataset}`, details ? el("span", { class: "muted small" }, " · " + details) : null),
      el("button", { type: "button", onclick: () => chooseVoice(clip, src) }, "Use this voice")),
    el("p", {}, clip.text),
    el("audio", { controls: true, preload: "none", src }));
}

function chooseVoice(clip, src) {
  state.voice = clip;
  $("chosen").hidden = false;
  $("chosen-name").textContent = `${voiceName(clip)} · ${clip.dataset}`;
  $("chosen-text").textContent = clip.text;
  $("chosen-audio").src = src;
  toast(`Voice chosen: ${voiceName(clip)}.`);
  if (mode === "generate") showTab("studio");          // step 2: the line and, optionally, a genome
}

$("voice-search").addEventListener("submit", (e) => { e.preventDefault(); loadVoices(1); });
$("dataset").addEventListener("change", () => loadVoices(1));
$("prev").addEventListener("click", () => loadVoices(state.page - 1));
$("next").addEventListener("click", () => loadVoices(state.page + 1));

function needVoice() {
  if (state.voice) return true;
  toast("Choose a voice first (the Choose a voice tab).", true);
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
    const starting = $("search-start");
    const selected = starting.value;
    starting.replaceChildren(el("option", { value: "" }, "Neutral"),
      ...named.map((genome) => el("option", { value: JSON.stringify(genome.genes) }, genome.name)));
    starting.value = [...starting.options].some((option) => option.value === selected) ? selected : "";
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
      el("li", { class: "item clickable", onclick: () => (n.run ? openRun(n.run, "ranked") : Studio.show("named:" + n.name)) },
        el("div", { class: "item-head" }, el("strong", {}, n.name),
          n.mos ? el("span", { class: "mos" }, "MOS " + fmt(n.mos.mos, 2)) : null),
        el("p", { class: "muted small" }, (n.run ? `Genome ${n.genome} of ${n.voice}` : "Made by hand in the studio") + (n.named_by ? ` · named by ${n.named_by}` : "")),
        el("p", { class: "markup" }, n.markup || ""),
        el("button", { type: "button", class: "quiet small", onclick: (e) => { e.stopPropagation(); Studio.show("named:" + n.name); } }, "Edit or render in the studio"))) : [el("li", { class: "muted" }, "None yet: rate a search, then pick its best genome.")]));
  } catch (error) { toast(error.message, true); }
}

$("pile-filter").addEventListener("change", loadSearches);
try { const saved = localStorage.getItem("pile"); if (saved) $("pile-filter").append(el("option", { value: saved }, saved)), $("pile-filter").value = saved; } catch (e) {}

$("search-method").addEventListener("change", () => {
  $("search-start").disabled = $("search-method").value === "random";
});

$("search-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!needVoice()) return;
  const body = { dataset: state.voice.dataset, clip_id: state.voice.clip_id, text: $("search-text").value,
                 genes: Number($("search-genes").value), method: $("search-method").value };
  if (body.method !== "random" && $("search-start").value) {
    body.starting_genome = JSON.parse($("search-start").value);
  }
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
  $("run-title").textContent = `${run.voice} · ${run.method_label || run.method} · ${run.genes_per_genome} genes · genome seed ${run.seed}`;
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
        // "fine" here is the automatic sanity check (words, pace, voice similarity), not a listener's verdict
        el("p", {}, `${Math.round(100 * s.shares.all.fine)} % of ${s.shares.all.judged} genomes passed the automatic sanity check ` +
          `(words, pace, voice similarity, measured by the machine; no one has listened to them for this). ` +
          `Works as a guard: ${s.verdicts.works_as_a_guard ? "yes" : "no"}. Beats the baseline: only blind listening can tell.`),
        el("div", { class: "table-wrap" }, el("table", {},
          el("thead", {}, el("tr", {}, ["Passed genomes minus baseline (measured)", "Mean", "95 % interval", "p"].map((h) => el("th", {}, h)))),
          el("tbody", {}, Object.entries(s.fine_minus_baseline).filter(([, d]) => d).map(([metric, d]) => el("tr", {},
            el("td", {}, metric), el("td", {}, fmt(d.mean)), el("td", {}, `${fmt(d.low)} to ${fmt(d.high)}`),
            el("td", {}, d.wilcoxon_p === null ? "–" : Number(d.wilcoxon_p).toPrecision(2))))))));
    }
    box.replaceChildren(...(parts.length ? parts : [el("p", { class: "muted" }, "No results on the host yet.")]));
  } catch (error) { box.replaceChildren(el("p", { class: "muted" }, error.message)); }
}

// ---------------------------------------------------------------- graphs --
// The six questions' charts as the host last made them (evidence-charts, or a question asked through the API)
async function loadStudy() {
  // The listening study from the piles each listener was assigned (GET /evidence/mos-study): totals only
  const box = $("mos-study");
  if (!box) return;
  box.replaceChildren(el("p", { class: "muted" }, "Loading…"));
  try {
    const s = await call("/evidence/mos-study");
    const r = s.report;
    if (!r) { box.replaceChildren(el("p", { class: "muted" }, s.report_note || "No study yet.")); return; }
    const pct = (x) => Math.round(100 * x) + " %";
    const p = (x) => x === null || x === undefined ? "–" : (x < 0.001 ? "< 0.001" : Number(x).toFixed(3));
    const conds = Object.values(r.conditions);
    box.replaceChildren(
      el("p", { class: "graphs-stamp" }, el("strong", {}, "Report evaluation: the three methods"),
        el("span", { class: "muted" }, ` · ${r.scores} scores of ${r.items} items on ${r.voices} test voices · assigned piles only (${r.listeners.join(", ")})`)),
      el("div", { class: "table-wrap" }, el("table", {},
        el("thead", {}, el("tr", {}, ["Method", "Mean", "95 % interval", "Scored 4 or more", "Scored 2 or less", "Artifact flags"].map((h) => el("th", {}, h)))),
        el("tbody", {}, conds.map((c) => el("tr", {}, el("td", {}, c.label), el("td", {}, fmt(c.mean, 2)),
          el("td", {}, `${fmt(c.ci95[0], 2)} – ${fmt(c.ci95[1], 2)}`), el("td", {}, pct(c.share_4_or_more)),
          el("td", {}, pct(c.share_2_or_less)), el("td", {}, String(c.artifact_flags))))))),
      el("div", { class: "table-wrap" }, el("table", {},
        el("thead", {}, el("tr", {}, ["Same item, same listener", "Mean difference", "95 % interval", "First higher / tie / second higher", "p (Holm)"].map((h) => el("th", {}, h)))),
        el("tbody", {}, r.paired.map((d) => el("tr", {}, el("td", {}, `${d.a} vs ${d.b}`),
          el("td", {}, (d.mean_difference > 0 ? "+" : "") + fmt(d.mean_difference, 2)),
          el("td", {}, `${fmt(d.ci95[0], 2)} to ${fmt(d.ci95[1], 2)}`),
          el("td", {}, `${d.a_higher} / ${d.tie} / ${d.b_higher}`), el("td", {}, p(d.holm_p))))))),
      el("div", { class: "graphs" }, s.charts.map((c) => el("figure", { class: "graph" },
        el("figcaption", {}, el("strong", {}, c.title)),
        el("a", { href: c.url, target: "_blank", rel: "noopener", title: "Open full size" },
          el("img", { src: c.url, alt: c.title, loading: "lazy" })),
        el("p", { class: "muted small" }, c.note)))),
      r.harm ? el("h3", {}, "Which genomes hurt (exploratory)") : null,
      r.harm ? el("p", { class: "muted small" }, `The worst 10 of ${r.harm.genomes} genomes carry ${Math.round(100 * r.harm.share_of_loss_in_worst_10)} % ` +
        `of the loss against the plain line; without the worst 15, the rest average ${fmt(r.harm.rest_without_worst["15"], 2)}. ` +
        `The automatic measures barely follow it (Spearman: words ${fmt(r.harm.measures_vs_listeners.wer, 2)}, ` +
        `voice ${fmt(r.harm.measures_vs_listeners.similarity, 2)}, pace ${fmt(r.harm.measures_vs_listeners.pace, 2)}); genomes a listener flagged ` +
        `for artifacts average ${fmt(r.harm.flagged.minus_plain, 2)}. ${r.harm.note}`) : null,
      r.harm ? el("div", { class: "table-wrap" }, el("table", {},
        el("thead", {}, el("tr", {}, ["Gene type", "Genomes", "Against plain", "Worse / better", "Allowed in guarded"].map((h) => el("th", {}, h)))),
        el("tbody", {}, r.gene_types.map((t) => el("tr", {}, el("td", {}, t.gene_type), el("td", {}, String(t.genomes)),
          el("td", {}, (t.minus_plain > 0 ? "+" : "") + fmt(t.minus_plain, 2)), el("td", {}, `${t.worse} / ${t.better}`),
          el("td", {}, t.in_guarded ? "yes" : "no")))))) : null,
      r.harm ? el("div", { class: "table-wrap" }, el("table", {},
        el("thead", {}, el("tr", {}, ["Where the events sit, or genes", "Genomes", "Against plain", "Worse / better"].map((h) => el("th", {}, h)))),
        el("tbody", {}, [...Object.entries(r.harm.positions), ...Object.entries(r.harm.gene_count).map(([n, v]) => [`${n} gene${n === "1" ? "" : "s"}`, v])]
          .map(([k, v]) => el("tr", {}, el("td", {}, k), el("td", {}, String(v.genomes)),
            el("td", {}, (v.minus_plain > 0 ? "+" : "") + fmt(v.minus_plain, 2)), el("td", {}, `${v.worse} / ${v.better}`)))))) : null,
      s.pilot ? el("p", { class: "muted small" }, `Pilot (mos-diverse-v1, ${s.pilot.genome_scores} genome scores, one listener a pile): ` +
        `genomes ${fmt(s.pilot.genome_mean, 2)} against the plain line ${fmt(s.pilot.plain_mean, 2)}; ` +
        `${s.pilot.above_plain} above the plain line, ${s.pilot.same_as_plain} the same, ${s.pilot.below_plain} below.`) : null,
      el("p", { class: "muted small" }, `Left out: ${Object.entries(s.scores_left_out).map(([k, v]) => `${k} (${v})`).join(", ") || "nothing"}.`));
  } catch (error) { box.replaceChildren(el("p", { class: "muted" }, error.message)); }
}

async function loadGraphs() {
  loadStudy();
  const box = $("graphs");
  box.replaceChildren(el("p", { class: "muted" }, "Loading…"));
  try {
    const g = await call("/evidence/charts/latest");
    const range = g.range && (g.range.from || g.range.to)
      ? `scores given from ${g.range.from || "the first score"} to ${g.range.to || "now"}` : "every score so far";
    const made = new Date(g.made.replace(" ", "T"));
    const when = isNaN(made) ? g.made : made.toLocaleString([], { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
    box.replaceChildren(
      el("p", { class: "graphs-stamp" }, el("strong", {}, `Made ${when}`),
        el("span", { class: "muted" }, ` · ${range} · ${g.genome_scores} genome scores, ${g.sanity_checks} sanity checks`)),
      el("div", { class: "graphs" }, g.charts.map((c) => el("figure", { class: "graph" },
        el("figcaption", {}, el("strong", {}, `${c.number}. ${c.question}`)),
        c.url ? el("a", { href: c.url, target: "_blank", rel: "noopener", title: "Open full size" },
          el("img", { src: c.url, alt: c.question, loading: "lazy" }))
          : el("div", { class: "no-data" }, el("strong", {}, "No data"), el("p", { class: "muted small" }, `${c.skipped || "nothing to draw"}.`)),
        // the text under a chart is Gemma 4's reading of its numbers (API calls/chart_captions.py), never ours
        c.url ? (c.caption
          ? el("div", { class: "caption" }, el("p", {}, c.caption.text),
              el("p", { class: "muted small" }, `Written by Gemma 4 (${c.caption.model || "Gemma"}) on ${c.caption.made || "?"}` +
                (c.caption.unverified_numbers ? ` · numbers not found in the data: ${c.caption.unverified_numbers.join(", ")}` : "")),
              c.caption.facts && c.caption.facts.length ? el("details", { class: "muted small" }, el("summary", {}, "What Gemma used"),
                el("ul", { class: "facts" }, c.caption.facts.map((f) => el("li", {}, el("code", {}, f.path), " = ", JSON.stringify(f.value))))) : null)
          : el("p", { class: "muted small" }, "Gemma 4 has not written the text for these numbers yet. It runs after python main_backend.py evidence-charts on the host."))
          : null,
        c.url && c.note ? el("details", { class: "muted small" }, el("summary", {}, "How to read this chart"), el("p", {}, c.note)) : null))),
      el("p", { class: "muted small" }, "To make them again from the newest scores, run python main_backend.py evidence-charts on the host."));
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
  setMode("home");                                    // every login opens on the home page
  const account = state.me && state.me.account;
  $("home-hello").textContent = `Welcome, ${state.me ? state.me.name : ""}`;
  const piles = (account && account.assignments) || [];
  $("home-rate-note").textContent = piles.length ? `You have ${piles.length} pile${piles.length === 1 ? "" : "s"} to rate.` : "";
  if (typeof Ask !== "undefined") Ask.init();         // Ask the Dramatizer and About, on the home page (ask.js)
}
connect().then((ok) => ok && start());
setInterval(() => { if (state.me) connect(); }, 30000);
