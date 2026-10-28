// The page: pick a dataset, filter its clips, click one to hear it and read its script.
// All data comes from the JSON routes in Voice_UI.py.

const $ = (id) => document.getElementById(id);

const state = { datasets: [], dataset: null, page: 1, pages: 1, clips: [], selected: null };

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}

function text(value) {
  return document.createTextNode(value ?? "");
}

function cell(className, content) {
  const td = document.createElement("td");
  if (className) td.className = className;
  td.append(text(content));
  return td;
}

// ----- datasets and filters -------------------------------------------------

function showTabs() {
  const tabs = $("tabs");
  tabs.replaceChildren();

  for (const dataset of state.datasets) {
    const button = document.createElement("button");
    button.type = "button";
    button.role = "tab";
    button.setAttribute("aria-selected", String(dataset === state.dataset));
    button.append(text(`${dataset.name} · ${dataset.clips.toLocaleString()}`));
    button.addEventListener("click", () => chooseDataset(dataset));
    tabs.append(button);
  }
}

function chooseDataset(dataset) {
  state.dataset = dataset;
  state.page = 1;
  showTabs();

  const speaker = $("speaker");
  speaker.replaceChildren(new Option("All speakers", ""));
  for (const [id, count] of Object.entries(dataset.speakers)) {
    speaker.append(new Option(`${id} (${count})`, id));
  }

  // One drop-down for each detail the dataset has: emotion, intensity, microphone...
  const holder = $("detail-filters");
  holder.replaceChildren();
  for (const [key, values] of Object.entries(dataset.filters)) {
    const label = document.createElement("label");
    label.append(text(key[0].toUpperCase() + key.slice(1)));
    const select = document.createElement("select");
    select.dataset.key = key;
    select.append(new Option(`Any ${key}`, ""));
    for (const value of values) select.append(new Option(value, value));
    select.addEventListener("change", reload);
    label.append(select);
    holder.append(label);
  }

  $("text").value = "";
  reload();
}

// ----- the clip list --------------------------------------------------------

function query() {
  const params = new URLSearchParams({ dataset: state.dataset.name, page: state.page });
  if ($("speaker").value) params.set("speaker", $("speaker").value);
  if ($("text").value.trim()) params.set("text", $("text").value.trim());
  for (const select of $("detail-filters").querySelectorAll("select")) {
    if (select.value) params.set(select.dataset.key, select.value);
  }
  return params;
}

function reload() {
  state.page = 1;
  load();
}

async function load() {
  const data = await getJSON(`/api/clips?${query()}`);
  state.page = data.page;
  state.pages = data.pages;
  state.clips = data.clips;

  $("count").textContent = `${data.total.toLocaleString()} clips`;
  $("page-label").textContent = `Page ${data.page} of ${data.pages}`;
  $("prev").disabled = data.page <= 1;
  $("next").disabled = data.page >= data.pages;

  const rows = $("rows");
  rows.replaceChildren();

  if (!data.clips.length) {
    const tr = document.createElement("tr");
    const td = cell("empty", "No clips match these filters.");
    td.colSpan = 3;
    tr.append(td);
    rows.append(tr);
    return;
  }

  data.clips.forEach((clip, index) => {
    const tr = document.createElement("tr");
    tr.tabIndex = 0;
    tr.setAttribute("aria-selected", String(state.selected?.id === clip.id));
    tr.append(
      cell("speaker", clip.speaker),
      cell("details", Object.values(clip.details).join(" · ")),
      cell("", clip.text),
    );
    tr.addEventListener("click", () => play(index));
    tr.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); play(index); }
    });
    rows.append(tr);
  });
}

// ----- the player -----------------------------------------------------------

async function play(index) {
  const clip = state.clips[index];
  if (!clip) return;

  state.selected = { ...clip, index };

  for (const [i, tr] of [...$("rows").children].entries()) {
    tr.setAttribute("aria-selected", String(i === index));
  }

  const info = await getJSON(`/api/clip/${encodeURIComponent(clip.dataset)}/${encodeURIComponent(clip.id)}`);

  $("placeholder").hidden = true;
  $("now").hidden = false;
  $("now-meta").textContent = `${info.dataset} · speaker ${info.speaker}`;
  $("now-script").textContent = info.text;
  $("now-file").textContent = info.file;

  const details = $("now-details");
  details.replaceChildren();
  for (const [key, value] of [["Clip", info.id], ...Object.entries(info.details)]) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.append(text(key[0].toUpperCase() + key.slice(1)));
    dd.append(text(value));
    details.append(dt, dd);
  }

  const audio = $("audio");
  audio.src = info.audio;
  audio.play().catch(() => {});   // the browser may need a click first; the controls are there

  // Other scripts on the page (the generate panel) can follow the chosen clip
  document.dispatchEvent(new CustomEvent("clip-selected", { detail: info }));
}

async function playNext() {
  const next = (state.selected?.index ?? -1) + 1;

  if (next < state.clips.length) {
    play(next);
  } else if (state.page < state.pages) {
    state.page += 1;
    await load();
    play(0);
  }
}

// ----- wiring ---------------------------------------------------------------

let typing;
$("text").addEventListener("input", () => { clearTimeout(typing); typing = setTimeout(reload, 250); });
$("speaker").addEventListener("change", reload);
$("filters").addEventListener("submit", (event) => event.preventDefault());
$("reset").addEventListener("click", () => chooseDataset(state.dataset));

$("prev").addEventListener("click", () => { state.page -= 1; load(); });
$("next").addEventListener("click", () => { state.page += 1; load(); });

$("replay").addEventListener("click", () => { const a = $("audio"); a.currentTime = 0; a.play(); });
$("play-next").addEventListener("click", playNext);
$("audio").addEventListener("ended", playNext);

$("copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("now-file").textContent);
    $("copy").textContent = "Copied";
    setTimeout(() => { $("copy").textContent = "Copy"; }, 1200);
  } catch {
    // Clipboard blocked: select the text so it can be copied by hand
    getSelection().selectAllChildren($("now-file"));
  }
});

getJSON("/api/datasets").then((datasets) => {
  state.datasets = datasets;
  chooseDataset(datasets[0]);
});
