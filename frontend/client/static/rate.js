// rate.js | The Rate page: one listener scores one listening pile, then checks their ranking.
// Uses app.js's helpers (call, el, $, toast, state, fmt). Scores are saved as soon as a slider is set,
// each with two answers: "did you hear an artifact?" and "did it sound like a different person than the
// original recording?" (the boxes under the slider).
//
// The pile is blind: the plain line is mixed in with the genomes, every clip is named by a token from the
// host, and each listener hears a voice's clips in their own shuffled order as Clip A, B, C... Nothing on the
// page says which clip is the plain line. Each voice opens with its original recording, to listen to first.
"use strict";

const Rate = (() => {
  const r = { piles: null, key: "", data: null, run: 0, view: "rate", pending: {}, timers: {}, player: new Audio(), playing: null };
  const STEPS = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5];
  const audioUrl = (run, clip) => `${API}/genome/searches/${encodeURIComponent(run)}/audio/${clip}.wav`;
  const referenceUrl = (run) => `${API}/genome/searches/${encodeURIComponent(run)}/reference.wav`;
  const store = (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} };
  const recall = (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } };
  // The two yes/no answers kept with a score: the field the host returns, and the box's text
  const ANSWERS = {
    artifact: { mine: "mine_artifact", draft: "draft_artifact", stored: "stored_artifacts", send: "artifacts",
                label: "I heard an artifact", hint: " (a glitch, click, noise, cut-off or garbled word)", tag: "artifact" },
    different: { mine: "mine_different", draft: "draft_different", stored: "stored_different", send: "different",
                 label: "Sounds like a different person", hint: " than the original recording", tag: "different person" },
  };

  // ------------------------------------------------------------ the pile --
  async function open() {
    try {
      const { piles, assigned } = await call("/genome/listening");
      r.piles = piles;
      const options = [];
      for (const [batch, ps] of Object.entries(piles)) for (const p of ps) {
        const mine = (assigned || []).includes(`${batch}|${p}`);
        options.push(el("option", { value: `${batch}|${p}` }, `${batch} · pile ${p}` + (mine ? " (yours)" : "")));
      }
      $("rp-pile").replaceChildren(el("option", { value: "" }, options.length ? "Choose your pile…" : "No piles on the host"), ...options);
      // Your own pile first; another pile only if you chose it last time and it is still yours to open
      const last = recall("rate-pile");
      const yours = (assigned || [])[0];
      const saved = yours && !(assigned || []).includes(last) && !r.chosen ? yours : last || yours || recall("pile");
      $("rp-pile").value = options.some((o) => o.value === saved) ? saved : "";
      await load();
    } catch (error) { toast(error.message, true); }
  }

  async function load() {
    r.key = $("rp-pile").value;
    store("rate-pile", r.key);
    const none = !r.key;
    $("rp-empty").hidden = !none;
    $("rp-empty").textContent = "Choose the pile the host gave you. Each voice has its original recording and a set of clips to score.";
    $("rp-rate").hidden = none || r.view !== "rate";
    $("rp-rank").hidden = none || r.view !== "rank";
    $("rp-progress").hidden = none;
    if (none) return;
    const [batch, pile] = r.key.split("|");
    try {
      r.data = await call("/genome/listening?" + new URLSearchParams({ batch, pile }));
    } catch (error) { toast(error.message, true); return; }
    const savedRun = Number(recall(`rate-run:${r.key}`));
    const firstOpen = r.data.runs.findIndex((x) => x.mine < x.clips);
    r.run = Number.isInteger(savedRun) && savedRun >= 0 && savedRun < r.data.runs.length ? savedRun : Math.max(0, firstOpen);
    $("rk-voice").replaceChildren(el("option", { value: "" }, "All voices"),
      ...r.data.runs.map((x) => el("option", { value: x.id }, x.voice)));
    render();
    resendUnsent();
  }

  function render() {
    progress();
    if (r.view === "rate") renderRun(); else renderRank();
  }

  function progress() {
    const done = r.data.runs.reduce((n, x) => n + x.mine, 0);
    const total = r.data.runs.reduce((n, x) => n + x.clips, 0);
    $("rp-fill").style.width = `${total ? (100 * done) / total : 0}%`;
    $("rp-progress-text").textContent = `${r.data.rater}: ${done} of ${total} clips scored` + (done === total && total ? " · all done, thank you" : "");
  }

  function setView(view) {
    r.view = view;
    document.querySelectorAll("[data-rview]").forEach((b) => b.classList.toggle("on", b.dataset.rview === view));
    $("rp-rate").hidden = !r.key || view !== "rate";
    $("rp-rank").hidden = !r.key || view !== "rank";
    if (r.data) render();
  }

  // ------------------------------------------------------- rating a run --
  function clipsOf(runId) { return r.data.clips.filter((c) => c.run === runId); }

  // The voice's original recording: the listener hears it first, to judge whether each clip is the same person
  const heardKey = (runId) => `rate-heard:${r.data.rater}:${runId}`;
  function referenceCard(run) {
    if (!run.reference)
      return el("div", { class: "reference missing" }, el("strong", {}, "Original recording"),
        el("p", { class: "muted small" }, "The host does not have this voice's original recording. Score the clips as usual and leave the person box empty."));
    const heard = recall(heardKey(run.id)) === "1";
    const note = el("span", { class: "small reference-note" + (heard ? " heard" : "") },
      heard ? "✓ Listened" : "Listen to this first");
    const player = el("audio", { controls: true, preload: "none", src: referenceUrl(run.id), onplay: pauseOthers,
                                 onended: () => { store(heardKey(run.id), "1"); note.textContent = "✓ Listened"; note.classList.add("heard"); } });
    return el("div", { class: "reference" },
      el("div", { class: "item-head" }, el("strong", {}, "Original recording"), note),
      el("p", { class: "muted small" }, "The real person this voice was cloned from, saying another line. It is not scored: use it to judge whether each clip sounds like the same person."),
      player);
  }

  function renderRun() {
    const runs = r.data.runs;
    $("rp-voices").replaceChildren(...runs.map((x, i) =>
      el("button", { type: "button", class: "step" + (i === r.run ? " on" : "") + (x.mine === x.clips ? " done" : ""),
                     "aria-current": i === r.run ? "step" : null, onclick: () => goTo(i) },
        el("span", {}, x.voice), el("small", {}, `${x.mine}/${x.clips}`))));
    const run = runs[r.run];
    const clips = clipsOf(run.id);
    $("rp-run").replaceChildren(
      el("h2", {}, run.voice), el("p", { class: "line" }, run.text),
      referenceCard(run),
      el("div", { class: "run-status" }, el("span", { id: "rp-run-count" }),
        el("button", { type: "button", class: "quiet small", id: "rp-next-todo", onclick: nextUnrated }, "Next unrated ›")),
      el("ol", { class: "clips rate-clips" }, clips.map((c) =>
        el("li", { class: `clip rate-clip ${stateOf(c)}`, id: `rc-${run.id}-${c.clip}` },
          el("div", { class: "item-head" }, el("strong", {}, `Clip ${c.label}`),
            el("span", { class: "rated-badge small", "aria-live": "polite" }, BADGES[stateOf(c)](c))),
          el("audio", { controls: true, preload: "none", src: audioUrl(run.id, c.clip), onplay: pauseOthers }),
          scoreSlider(c), answerBox(c, "artifact"), run.reference ? answerBox(c, "different") : null))));
    $("rp-prev").disabled = r.run === 0;
    $("rp-next").textContent = r.run === runs.length - 1 ? "See my ranking ›" : "Next voice ›";
    runCount(run.id);
  }

  // "7 of 11 rated in this voice · 4 to go", and whether anything in the pile is left
  function runCount(runId) {
    const out = $("rp-run-count");
    if (!out || r.data.runs[r.run].id !== runId) return;
    const clips = clipsOf(runId), done = clips.filter((c) => c.mine !== null).length;
    out.textContent = done === clips.length ? `✓ All ${clips.length} rated in this voice`
      : `${done} of ${clips.length} rated in this voice · ${clips.length - done} to go`;
    out.className = done === clips.length ? "all-rated" : "";
    $("rp-next-todo").hidden = !r.data.clips.some((c) => c.mine === null);
  }

  // The first clip nobody has scored yet, from this voice on (then from the start of the pile)
  function nextUnrated() {
    const order = r.data.runs.slice(r.run).concat(r.data.runs.slice(0, r.run));
    for (const run of order) {
      const c = clipsOf(run.id).find((x) => x.mine === null && !r.pending[x.run]?.[x.clip]);
      if (!c) continue;
      const i = r.data.runs.indexOf(run);
      if (i !== r.run) goTo(i);
      const box = card(c);
      box.scrollIntoView({ block: "center", behavior: "smooth" });
      box.classList.add("flash");
      setTimeout(() => box.classList.remove("flash"), 1500);
      return;
    }
    toast("Everything in this pile is rated.");
  }

  // Ask the host again what it has kept, and show that: any score this page thinks is saved but
  // the host does not have is shown as not rated, so it can be set again.
  async function checkWithHost() {
    const button = $("rp-check");
    button.disabled = true;
    try {
      flushAll();
      const [batch, pile] = r.key.split("|");
      const fresh = await call("/genome/listening?" + new URLSearchParams({ batch, pile }));
      const host = new Map(fresh.clips.map((c) => [`${c.run}/${c.clip}`, c]));
      let differ = 0;
      for (const c of r.data.clips) {
        const h = host.get(`${c.run}/${c.clip}`);
        if (!h || r.pending[c.run]?.[c.clip]) continue;
        if (h.mine !== c.mine || Object.values(ANSWERS).some((a) => (h[a.mine] ?? null) !== (c[a.mine] ?? null))) differ++;
        for (const a of Object.values(ANSWERS)) c[a.draft] = h[a.mine] === true;
        Object.assign(c, { mine: h.mine, mine_artifact: h.mine_artifact, mine_different: h.mine_different, mine_on: h.mine_on });
      }
      r.data.runs.forEach((run) => recount(run.id));
      render();
      const done = fresh.clips.filter((c) => c.mine !== null).length;
      const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      $("rp-checked").textContent = `Host checked ${time}: ${done} of ${fresh.clips.length} rated` +
        (differ ? ` · ${differ} corrected` : " · all match");
      toast(differ ? `${differ} clip(s) differed from the host and now show what the host has.` : "The host has every score shown here.", !!differ);
    } catch (error) { toast("Could not check: " + error.message, true); }
    finally { button.disabled = false; }
  }

  function goTo(i) {
    flushAll();
    r.run = i;
    store(`rate-run:${r.key}`, String(i));
    renderRun();
    $("rp-voices").scrollIntoView({ block: "nearest" });
  }

  function pauseOthers(e) {
    document.querySelectorAll("#tab-rate audio").forEach((a) => { if (a !== e.target) a.pause(); });
    stopPlayer();
  }

  // A 0-5 slider in steps of 0.5. It counts only once it is set: an unset slider shows no value.
  function scoreSlider(c) {
    const id = `s-${c.run}-${c.clip}`;
    const out = el("output", { for: id, class: "score-value" }, c.mine === null ? "not scored" : c.mine.toFixed(1));
    const range = el("input", { type: "range", id, min: 0, max: 5, step: 0.5, value: c.mine === null ? 2.5 : c.mine,
                                "aria-label": `Score of clip ${c.label} from 0 to 5`,
                                "aria-valuetext": c.mine === null ? "not scored" : `${c.mine} of 5` });
    const box = el("div", { class: "score" + (c.mine === null ? " unset" : "") },
      el("span", { class: "end small" }, "0 bad"), range, el("span", { class: "end small" }, "5 excellent"), out,
      el("div", { class: "ticks", "aria-hidden": "true" }, STEPS.map((v) => el("span", { class: Number.isInteger(v) ? "major" : "" }, Number.isInteger(v) ? String(v) : ""))));
    const set = () => {
      const value = Number(range.value);
      box.classList.remove("unset");
      out.textContent = value.toFixed(1);
      range.setAttribute("aria-valuetext", `${value} of 5`);
      queueSave(c, value);
    };
    range.addEventListener("input", set);
    range.addEventListener("change", set);
    range.addEventListener("pointerup", set);       // a click on the unset slider's own value still counts
    range.addEventListener("keyup", (e) => { if (e.key === "Enter" || e.key === " ") set(); });
    return box;
  }

  // A yes/no answer kept with the score: ticked = yes, left empty = no. It is saved with the score, so
  // every score given from now on has an answer; ticking it on a scored clip saves it at once.
  function answerBox(c, kind) {
    const a = ANSWERS[kind];
    const id = `${kind[0]}-${c.run}-${c.clip}`;
    if (c[a.draft] === undefined) c[a.draft] = c[a.mine] === true;
    const box = el("input", { type: "checkbox", id, class: "artifact-check" });
    box.checked = c[a.draft];
    const note = el("span", { class: `answer-note muted small` }, answerNote(c, kind));
    box.addEventListener("change", () => {
      c[a.draft] = box.checked;
      const waiting = r.pending[c.run]?.[c.clip];
      const value = waiting ? waiting.value : c.mine;
      if (value !== null && value !== undefined) queueSave(c, value);
      note.textContent = answerNote(c, kind);
    });
    return el("div", { class: `artifact answer-${kind}` + (c[a.draft] ? " heard" : ""), "data-kind": kind },
      box, el("label", { for: id }, a.label), el("span", { class: "muted small" }, a.hint), note);
  }

  function answerNote(c, kind) {
    const mine = c[ANSWERS[kind].mine];
    if (c.mine === null && !r.pending[c.run]?.[c.clip]) return " · saved with the score";
    if (mine === null || mine === undefined) return c.mine !== null ? " · not answered yet: tick it, or set the score again for no" : "";
    return "";
  }

  function queueSave(c, value) {
    const answers = {};
    for (const [kind, a] of Object.entries(ANSWERS)) {
      answers[kind] = !!c[a.draft];
      const box = card(c)?.querySelector(`.answer-${kind}`);
      if (box) box.classList.toggle("heard", answers[kind]);
    }
    const reference = r.data.runs.find((x) => x.id === c.run)?.reference;
    if (!reference) delete answers.different;           // not asked without the original to compare with
    const same = Object.entries(answers).every(([kind, v]) => c[ANSWERS[kind].mine] === v);
    if (c.mine === value && same && !r.pending[c.run]?.[c.clip]) return;
    (r.pending[c.run] = r.pending[c.run] || {})[c.clip] = { value, answers, c };
    rememberUnsent();
    mark(c, "saving");
    clearTimeout(r.timers[c.run]);
    r.timers[c.run] = setTimeout(() => flush(c.run), 500);
  }

  // Scores not yet confirmed by the host are also kept in this browser, so a closed tab,
  // a lost connection or an expired login loses nothing: they are sent again later.
  const unsentKey = () => `rate-unsent:${r.data ? r.data.rater : ""}`;
  function rememberUnsent() {
    const out = {};
    for (const [run, items] of Object.entries(r.pending))
      for (const [clip, x] of Object.entries(items)) (out[run] = out[run] || {})[clip] = { value: x.value, answers: x.answers };
    store(unsentKey(), JSON.stringify(out));
  }

  const card = (c) => document.getElementById(`rc-${c.run}-${c.clip}`);
  function when(stamp) {           // "2026-11-10 14:03:12" -> "10 Nov, 14:03"
    const d = new Date(stamp.replace(" ", "T"));
    return isNaN(d) ? stamp : d.toLocaleString([], { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
  }

  // Whether a clip is rated, as the host last confirmed it: one badge and one colour per card.
  //   rated   ✓ the host has the score (with its date)      todo    ○ nobody scored it yet
  //   saving  the score is on its way                       failed  ! not saved yet, being sent again
  const tags = (c) => Object.values(ANSWERS).filter((a) => c[a.mine]).map((a) => ` · ${a.tag}`).join("");
  const BADGES = {
    rated: (c) => `✓ Rated ${c.mine.toFixed(1)}` + tags(c) + (c.mine_on ? ` · ${when(c.mine_on)}` : ""),
    todo: () => "○ Not rated yet",
    saving: () => "Saving…",
    failed: () => "! Not saved yet: trying again…",
  };
  const stateOf = (c) => (c.mine === null ? "todo" : "rated");
  function mark(c, how = stateOf(c)) {
    const box = card(c);
    if (!box) return;
    box.classList.remove("rated", "todo", "saving", "failed");
    box.classList.add(how);
    box.querySelector(".rated-badge").textContent = BADGES[how](c);
  }

  async function flush(runId, keepalive = false) {
    clearTimeout(r.timers[runId]);
    const items = r.pending[runId];
    delete r.pending[runId];
    if (!items || !Object.keys(items).length) return;
    const rater = r.data.rater;
    const body = { scores: { [rater]: Object.fromEntries(Object.entries(items).map(([k, x]) => [k, x.value])) } };
    for (const [kind, a] of Object.entries(ANSWERS))
      body[a.send] = { [rater]: Object.fromEntries(Object.entries(items)
        .filter(([, x]) => typeof x.answers?.[kind] === "boolean").map(([k, x]) => [k, x.answers[kind]])) };
    try {
      const reply = await call(`/genome/searches/${encodeURIComponent(runId)}/ratings`, { body, keepalive });
      const stored = (reply.stored || {})[rater] || {};
      const dates = (reply.rated_on || {})[rater] || {};
      for (const [clip, x] of Object.entries(items)) {
        if (stored[clip] !== x.value) throw new Error("the host did not keep every score");
        for (const [kind, a] of Object.entries(ANSWERS)) {
          if (typeof x.answers?.[kind] !== "boolean") continue;
          if (((reply[a.stored] || {})[rater] || {})[clip] !== x.answers[kind]) throw new Error("the host did not keep every answer");
          x.c[a.mine] = x.answers[kind];
        }
        x.c.mine = x.value;
        for (const kind of Object.keys(ANSWERS)) {
          const note = card(x.c)?.querySelector(`.answer-${kind} .answer-note`);
          if (note) note.textContent = answerNote(x.c, kind);
        }
        x.c.mine_on = dates[clip] || x.c.mine_on;
        if (!r.pending[runId]?.[clip]) mark(x.c, "rated");
      }
      recount(runId);
    } catch (error) {
      // Put them back (unless a newer score came meanwhile) and try again in a few seconds
      const back = (r.pending[runId] = r.pending[runId] || {});
      for (const [clip, x] of Object.entries(items)) {
        if (!back[clip]) back[clip] = x;
        mark(x.c, "failed");
      }
      if (!keepalive) toast("Not saved yet: " + error.message + ". It is kept here and sent again.", true);
      clearTimeout(r.timers[runId]);
      if (state.me) r.timers[runId] = setTimeout(() => flush(runId), 5000);
    } finally {
      rememberUnsent();
    }
  }

  function flushAll(keepalive = false) { Object.keys(r.pending).forEach((run) => flush(run, keepalive)); }

  // Scores left over from last time (the tab closed before the host answered): send them now.
  // Clips are named by the host's blind tokens; scores kept under an older page's names are dropped.
  function resendUnsent() {
    let left = {};
    try { left = JSON.parse(recall(unsentKey()) || "{}"); } catch (e) {}
    let n = 0;
    for (const [run, clips] of Object.entries(left)) for (const [clip, saved] of Object.entries(clips)) {
      const c = r.data.clips.find((x) => x.run === run && x.clip === clip);
      if (!c || !saved || typeof saved.value !== "number") continue;
      const answers = saved.answers || (typeof saved.artifact === "boolean" ? { artifact: saved.artifact } : {});
      for (const [kind, v] of Object.entries(answers)) if (ANSWERS[kind] && typeof v === "boolean") c[ANSWERS[kind].draft] = v;
      (r.pending[run] = r.pending[run] || {})[clip] = { value: saved.value, answers, c };
      n++;
    }
    if (n) { toast(`Sending ${n} score(s) that were not saved last time…`); flushAll(); }
    else rememberUnsent();
  }

  function recount(runId) {
    const run = r.data.runs.find((x) => x.id === runId);
    const clips = clipsOf(runId);
    run.mine = clips.filter((c) => c.mine !== null).length;
    progress();
    const step = $("rp-voices").children[r.data.runs.indexOf(run)];
    if (step) {
      step.querySelector("small").textContent = `${run.mine}/${run.clips}`;
      step.classList.toggle("done", run.mine === run.clips);
    }
    runCount(runId);
  }

  // ------------------------------------------------------------ ranking --
  // Your own scores, best first, by the same blind clip letters as on the Rate view
  function renderRank() {
    const voice = $("rk-voice").value;
    const clips = r.data.clips.filter((c) => !voice || c.run === voice);
    const scored = clips.filter((c) => c.mine !== null).sort((a, b) => b.mine - a.mine);
    const unscored = clips.filter((c) => c.mine === null);
    $("rk-head").textContent = scored.length ? `Your scores, ${r.data.rater}. Equal scores share a place.`
      : "Nothing scored yet: score some clips under Rate first.";
    let rank = 0, last = null;
    const rows = scored.map((c, i) => {
      if (c.mine !== last) { rank = i + 1; last = c.mine; }
      return rankRow(c, rank);
    });
    if (unscored.length) rows.push(el("li", { class: "rank-gap muted small" }, `Not scored yet (${unscored.length})`),
      ...unscored.map((c) => rankRow(c, null)));
    $("rk-list").replaceChildren(...rows);
  }

  function rankRow(c, rank) {
    const label = `Clip ${c.label}`;
    const v = c.mine;
    const bar = v === null ? el("div", { class: "bar empty" })
      : el("div", { class: "bar" }, el("span", { class: "fill", style: `width:${(v / 5) * 100}%` }));
    const shown = v === null ? "–" : v.toFixed(1);
    const tip = [c.voice, label, v === null ? "not scored" : shown, c.mine_on ? `scored ${when(c.mine_on)}` : null].filter(Boolean).join(" · ");
    const playing = r.playing === `${c.run}/${c.clip}`;
    return el("li", { class: "rank-row" + (v === null ? " unrated" : ""), title: tip },
      el("span", { class: "rank-no" }, rank ? `#${rank}` : ""),
      el("button", { type: "button", class: "play" + (playing ? " on" : ""), "aria-label": `Play ${label} of ${c.voice}`,
                     onclick: (e) => play(c, e.currentTarget) }, playing ? "❚❚" : "▶"),
      el("div", { class: "rank-label" }, el("strong", {}, label), el("span", { class: "muted small" }, ` · ${c.voice}`),
        ...Object.values(ANSWERS).filter((a) => c[a.mine]).map((a) => el("span", { class: "artifact-tag small" }, ` ${a.tag}`)),
        c.mine_on ? el("div", { class: "muted small" }, `scored ${when(c.mine_on)}`) : null),
      bar,
      el("span", { class: "rank-value" }, shown),
      el("button", { type: "button", class: "quiet small", onclick: () => rerate(c) }, "Re-score"));
  }

  function stopPlayer() {
    r.player.pause();
    r.playing = null;
    document.querySelectorAll("#rk-list .play.on").forEach((b) => { b.classList.remove("on"); b.textContent = "▶"; });
  }

  function play(c, button) {
    const id = `${c.run}/${c.clip}`;
    if (r.playing === id) return stopPlayer();
    stopPlayer();
    document.querySelectorAll("#tab-rate audio").forEach((a) => a.pause());
    r.player.src = audioUrl(c.run, c.clip);
    r.player.play().catch((error) => toast("Cannot play: " + error.message, true));
    r.playing = id;
    button.classList.add("on");
    button.textContent = "❚❚";
  }
  r.player.addEventListener("ended", stopPlayer);

  function rerate(c) {
    stopPlayer();
    r.run = r.data.runs.findIndex((x) => x.id === c.run);
    store(`rate-run:${r.key}`, String(r.run));
    setView("rate");
    const box = document.getElementById(`rc-${c.run}-${c.clip}`);
    if (box) {
      box.scrollIntoView({ block: "center", behavior: "smooth" });
      box.classList.add("flash");
      setTimeout(() => box.classList.remove("flash"), 1800);
      box.querySelector("input[type=range]").focus({ preventScroll: true });
    }
  }

  // ------------------------------------------------------------- wiring --
  $("rp-pile").addEventListener("change", () => { r.chosen = true; $("rp-checked").textContent = ""; flushAll(); load(); });
  $("rp-check").addEventListener("click", checkWithHost);
  document.querySelectorAll("[data-rview]").forEach((b) => b.addEventListener("click", () => { flushAll(); setView(b.dataset.rview); }));
  $("rp-prev").addEventListener("click", () => r.run > 0 && goTo(r.run - 1));
  $("rp-next").addEventListener("click", () => (r.run < r.data.runs.length - 1 ? goTo(r.run + 1) : (flushAll(), setView("rank"))));
  ["rk-by", "rk-voice"].forEach((id) => $(id).addEventListener("change", renderRank));
  // Closing or hiding the tab: send what is waiting at once, in a request the browser finishes even after the tab is gone
  window.addEventListener("pagehide", () => flushAll(true));
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden") flushAll(true); });
  window.addEventListener("beforeunload", (e) => {
    if (Object.keys(r.pending).length) { flushAll(true); e.preventDefault(); e.returnValue = ""; }
  });

  return { open };
})();
