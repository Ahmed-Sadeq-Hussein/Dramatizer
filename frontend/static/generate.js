// The generate panel: the clip picked in the list is the reference voice; the form
// starts a job on the server and follows it until the new audio is ready.
// app.js announces the picked clip with a "clip-selected" event.

let reference = null;     // the clip picked in the list
let polling = null;

const nameOf = (id) => document.getElementById(id);

function say(message, kind = "") {
  const status = nameOf("status");
  status.hidden = !message;
  status.textContent = message;
  status.className = `status ${kind}`;
}

function setBusy(busy) {
  nameOf("generate").disabled = busy || !reference;
  nameOf("generate").textContent = busy ? "Working…" : "Generate";
}

// ----- following the picked clip -------------------------------------------

document.addEventListener("clip-selected", (event) => {
  reference = event.detail;
  nameOf("generate").disabled = false;
  nameOf("voice-name").placeholder = `e.g. ${reference.speaker}`;

  const note = nameOf("reference-note");
  note.textContent = `Reference: ${reference.dataset} ${reference.speaker} · "${reference.text}"`;

  // Cloning works best with a few seconds of speech; most library clips are short
  const audio = nameOf("audio");
  audio.addEventListener("loadedmetadata", () => {
    if (reference && audio.duration < 3) {
      note.textContent += ` (${audio.duration.toFixed(1)} s: short clips clone less well)`;
    }
  }, { once: true });
});

// ----- starting a job and following it -------------------------------------

nameOf("generate-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!reference) return;

  setBusy(true);
  nameOf("result").hidden = true;
  say("Starting…");

  const response = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset: reference.dataset,
      clip_id: reference.id,
      text: nameOf("target").value,
      name: nameOf("voice-name").value,
    }),
  });
  const body = await response.json();

  if (!response.ok) {
    say(body.error || "Could not start.", "error");
    setBusy(false);
    return;
  }

  follow(body.id);
});

function follow(jobId) {
  clearInterval(polling);

  polling = setInterval(async () => {
    const response = await fetch(`/api/job/${jobId}`);
    if (!response.ok) { clearInterval(polling); say("Lost track of the job.", "error"); setBusy(false); return; }

    const job = await response.json();

    if (job.status === "done") {
      clearInterval(polling);
      showResult(job.result);
      say("");
      setBusy(false);
      loadRecent();
    } else if (job.status === "error") {
      clearInterval(polling);
      say(job.error, "error");
      setBusy(false);
    } else {
      const extra = job.status === "loading model" ? " (the first run takes about a minute)" : "";
      say(`${job.status[0].toUpperCase()}${job.status.slice(1)}${extra} · ${job.waited} s`);
    }
  }, 1000);
}

function showResult(result) {
  nameOf("result").hidden = false;
  nameOf("result-meta").textContent = `${result.voice} · ${result.seconds ?? "?"} s to generate`;

  const audio = nameOf("result-audio");
  audio.src = result.audio;
  audio.play().catch(() => {});
}

// ----- recent results -------------------------------------------------------

async function loadRecent() {
  const items = await (await fetch("/api/generated")).json();
  const list = nameOf("recent");
  list.replaceChildren();

  if (!items.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "Nothing generated yet.";
    list.append(li);
    return;
  }

  for (const item of items) {
    const li = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "quiet";
    button.textContent = item.voice;
    button.addEventListener("click", () => showResult({ ...item, seconds: item.seconds }));

    const text = document.createElement("span");
    text.textContent = item.text || item.file;

    li.append(button, text);
    list.append(li);
  }
}

loadRecent();
