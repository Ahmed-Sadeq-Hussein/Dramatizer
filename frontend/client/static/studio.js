// studio.js | The genome studio: pick a genome, edit its genes, save it under a name, render it in the chosen voice.
// Uses the helpers of app.js ($, el, call, toast, state, can, needVoice, followJob).
"use strict";

const Studio = (() => {
  const S = { lib: null, options: null, genes: [], picked: null, changed: false, timer: null, words: [] };
  const label = (f) => f.replace(/_/g, " ");

  async function open() {
    showVoice();
    const key = S.pending;
    S.pending = null;
    if (!S.lib) await load(key);
    else if (key) showList(key);
    if (S.incoming) take();
  }

  // Genes from elsewhere (a render's Open in studio, or Paste genome): edited here as a new, unsaved genome
  function useGenes(genes, about = {}) {
    S.incoming = { genes, ...about };
    if (!$("tab-studio").hidden && S.lib) take();
    else showTab("studio");            // calls open(), which takes them once the library is loaded
  }

  async function take() {
    const { genes, from, text, voice } = S.incoming;
    S.incoming = null;
    S.picked = null;
    S.from = from || "the clipboard";
    S.genes = genes.map((g) => ({ ...g }));
    S.changed = true;
    $("st-pick").value = "";
    $("st-remove").hidden = true;
    blank(false);
    $("st-picked").textContent = `Copied from ${from || "the clipboard"}: edit it, render it, or save it under a name.`;
    if (text) $("st-text").value = text;
    renderGenes();
    schedulePreview();
    if (voice && !(state.voice && state.voice.dataset === voice.dataset && state.voice.clip_id === voice.clip_id)) {
      try {                             // the render's own voice, when this account may use it
        const { clips } = await call("/voices?" + new URLSearchParams({ dataset: voice.dataset, q: voice.speaker }));
        const clip = clips.find((c) => c.clip_id === voice.clip_id);
        if (clip) {
          chooseVoice(clip, `${API}/voices/audio?` + new URLSearchParams({ dataset: clip.dataset, clip_id: clip.clip_id }));
          showVoice();
        }
      } catch { /* the voice stays as it was */ }
    }
  }

  async function paste() {
    let text = null;
    try { text = await navigator.clipboard.readText(); } catch { /* not allowed here: ask for it */ }
    if (!text || !text.trim().startsWith("{") && !text.trim().startsWith("[")) {
      text = window.prompt("Paste a copied genome (Copy genome on a render, or in the studio):", "");
      if (text === null) return;
    }
    try {
      const found = readGenome(text);
      useGenes(found.genes, { from: found.from || "the clipboard" });
      toast(`Pasted ${found.genes.length} gene${found.genes.length === 1 ? "" : "s"}.`);
    } catch (error) { toast(error.message, true); }
  }

  function showVoice() {
    $("st-voice").textContent = state.voice ? `Voice: ${state.voice.speaker} · ${state.voice.dataset}`
                                            : "No voice chosen yet: choose one in step 1, Voices.";
  }

  async function load(selectKey) {
    try {
      const data = await call("/genome/library");
      S.lib = data.genomes;
      S.options = data.options;
      showList(selectKey);
      if (!S.genes.length && !S.picked) { renderGenes(); schedulePreview(); }
    } catch (error) { toast(error.message, true); }
  }

  // ------------------------------------------------------------ picking --
  function title(g) {
    const what = (g.described && g.described.length ? g.described : g.genes.map((x) => label(x.feature))).join(", ");
    const mos = (g.mos !== null && g.mos !== undefined ? ` · MOS ${Number(g.mos).toFixed(2)}` : "") +
      (g.unusable ? " · can't render as it is" : "");
    if (g.source === "search") return `#${g.genome}${mos} — ${what}`;
    if (g.source === "private") return `● ${g.name} — ${what}`;
    return `${g.source === "edited" ? "✎" : "★"} ${g.name}${mos} — ${what}`;
  }

  function matches(g, find) {
    if (!find) return true;
    const hay = [g.name, g.voice, g.text, g.run, (g.described || []).join(" "), g.genes.map((x) => x.feature).join(" ")]
      .join(" ").toLowerCase().replace(/_/g, " ");
    return find.toLowerCase().split(/\s+/).every((w) => hay.includes(w));
  }

  function showList(selectKey) {
    const source = $("st-source").value, find = $("st-find").value.trim();
    const shown = S.lib.filter((g) => (!source || g.source === source) && matches(g, find));
    const groups = new Map();
    const first = shown.slice(0, 400);
    const wanted = selectKey && S.lib.find((g) => g.key === selectKey);
    if (wanted && !first.includes(wanted)) first.unshift(wanted);          // opened from a link (Ask)
    for (const g of first) {
      if (!groups.has(g.group)) groups.set(g.group, el("optgroup", { label: g.group || "Other" }));
      groups.get(g.group).append(el("option", { value: g.key, title: g.markup || "" }, title(g)));
    }
    $("st-pick").replaceChildren(...groups.values());
    if (shown.length > 400) $("st-pick").append(el("option", { disabled: true }, `… ${shown.length - 400} more: narrow the search`));
    if (!shown.length) $("st-pick").append(el("option", { disabled: true }, "Nothing matches."));
    if (selectKey && wanted) { $("st-pick").value = selectKey; pick(selectKey); }
    else if (selectKey) toast("That genome is not in your genome library (your account may not see it).", true);
  }

  function pick(key) {
    const g = S.lib.find((x) => x.key === key);
    if (!g) return;
    S.picked = g;
    S.changed = false;
    S.genes = g.genes.map((x) => ({ ...x }));
    const from = g.source === "search" ? `Genome ${g.genome} of the search ${g.run}` :
      g.source === "edited" ? `Made by hand${g.named_by ? " by " + g.named_by : ""}${g.based_on ? ", from " + g.based_on : ""}`
                            : `Named${g.named_by ? " by " + g.named_by : ""}, genome ${g.genome} of ${g.voice}`;
    const parts = [from + (g.verdict ? ` · sanity: ${g.verdict}` : "") +
      (g.unusable ? ` · ${g.unusable}: change or remove that gene to render it` : "")];
    $("st-picked").replaceChildren(parts.join(""), g.text ? " · " : "",
      g.text ? el("button", { type: "button", class: "quiet small", onclick: () => { $("st-text").value = g.text; schedulePreview(); } },
                  "Use its own line") : "");
    $("st-remove").hidden = !(g.source === "edited" && can("name"));
    blank(g.source === "private", g.source === "private" ? g.name : "");
    renderGenes();
    schedulePreview();
  }

  // A blank genome (or one of your own) gets a name, and is kept as yours when rendered
  function blank(on, name = "") {
    $("st-blank").hidden = $("st-blank-note").hidden = !on;
    $("st-blank-name").value = name;
  }

  // ------------------------------------------------------------ editing --
  function changed() {
    S.changed = true;
    renderGenes();
    schedulePreview();
  }

  function geneRow(gene, i) {
    const hasSpeed = S.genes.some((x, j) => j !== i && x.feature === "speed");
    const feature = el("select", { "aria-label": `Gene ${i + 1} type`, onchange: (e) => {
        const f = e.target.value;
        if (f === "speed") S.genes[i] = { feature: "speed", rate: 1.0 };
        else {
          S.genes[i] = { feature: f, position: gene.position ?? 0.5 };
          if (f === "pause") S.genes[i].kind = gene.kind || S.options.pause_kinds[0];
        }
        changed();
      } },
      ...S.options.features.map((f) => {
        const off = (S.options.disabled || []).includes(f);
        return el("option", { value: f, disabled: (f === "speed" && hasSpeed) || off }, label(f) + (off ? " (disabled)" : ""));
      }));
    feature.value = gene.feature;

    let where;
    if (gene.feature === "speed") {
      const [lo, hi] = S.options.speed_bounds;
      where = el("div", { class: "where" },
        el("input", { type: "number", min: lo, max: hi, step: 0.05, value: gene.rate, "aria-label": `Gene ${i + 1} speed`,
                      onchange: (e) => { const v = Number(e.target.value); if (v >= lo && v <= hi) { gene.rate = v; changed(); } else toast(`Speed is from ${lo} to ${hi}.`, true); } }),
        el("span", { class: "muted small" }, "× speed"));
    } else {
      const value = el("span", { class: "muted small" }, Number(gene.position).toFixed(2));
      where = el("div", { class: "where" },
        el("input", { type: "range", min: 0, max: 1, step: 0.01, value: gene.position, "aria-label": `Gene ${i + 1} position`,
                      oninput: (e) => { gene.position = Number(e.target.value); value.textContent = gene.position.toFixed(2); S.changed = true; schedulePreview(); } }),
        gene.feature === "pause"
          ? el("select", { "aria-label": `Gene ${i + 1} pause kind`, onchange: (e) => { gene.kind = e.target.value; changed(); } },
              ...S.options.pause_kinds.map((k) => el("option", { value: k, selected: k === gene.kind }, k)))
          : value,
        el("span", { class: "word", id: `st-word-${i}` }, S.words[i] || ""));
    }
    return el("li", { class: "gene" }, el("span", { class: "num" }, i + 1), feature, where,
      el("button", { type: "button", class: "quiet remove", "aria-label": `Remove gene ${i + 1}`,
                     onclick: () => { S.genes.splice(i, 1); changed(); } }, "×"));
  }

  function renderGenes() {
    $("st-genes").replaceChildren(...S.genes.map(geneRow));
    const max = S.options ? S.options.max_genes : 10;
    $("st-count").textContent = S.genes.length ? `${S.genes.length} of at most ${max} genes` : "No genes: this is the plain line.";
    $("st-add").disabled = S.genes.length >= max;
  }

  function schedulePreview() {
    clearTimeout(S.timer);
    S.timer = setTimeout(preview, 250);
  }

  async function preview() {
    const text = $("st-text").value.trim();
    if (!text) { $("st-markup").textContent = "Type a line."; return; }
    const asked = (S.asked = (S.asked || 0) + 1);
    try {
      const { preview: p } = await call("/genome/preview", { body: { text, genes: S.genes } });
      if (asked !== S.asked) return;          // a newer preview was asked for: an older answer must not overwrite it
      S.words = p.described;
      p.described.forEach((d, i) => { const w = $(`st-word-${i}`); if (w) w.textContent = d; });
      $("st-markup").textContent = p.markup;
      $("st-speed").textContent = p.speed !== 1 ? `Speed ${p.speed}` : "";
      $("st-warnings").replaceChildren(...p.warnings.map((w) => el("li", {}, w)));
    } catch (error) {
      if (asked === S.asked) $("st-markup").textContent = error.message;
    }
  }

  // ------------------------------------------------------ save, render --
  async function save() {
    const name = $("st-name").value.trim();
    if (!name) { toast("Give the genome a name.", true); return; }
    if (!S.genes.length) { toast("Add at least one gene.", true); return; }
    try {
      const body = { name, genes: S.genes, text: $("st-text").value.trim() };
      if (S.picked) body.based_on = S.picked.name || `${S.picked.voice} genome ${S.picked.genome} (${S.picked.run})`;
      const { named } = await call("/genome/edited", { body });
      toast(`Saved as “${named.name}”.`);
      $("st-name").value = "";
      await load("named:" + named.name);
    } catch (error) { toast(error.message, true); }
  }

  async function remove() {
    const g = S.picked;
    if (!g || g.source !== "edited") return;
    if (!confirm(`Delete the genome “${g.name}”? This cannot be undone.`)) return;
    try {
      await call("/genome/edited/remove", { body: { name: g.name } });
      toast(`Deleted “${g.name}”.`);
      S.picked = null;
      $("st-picked").textContent = "";
      $("st-remove").hidden = true;
      await load();
    } catch (error) { toast(error.message, true); }
  }

  async function renderOne(genes, what, privateName) {
    const body = { dataset: state.voice.dataset, clip_id: state.voice.clip_id, text: $("st-text").value.trim(), genes, name: what };
    if (privateName) body.private_name = privateName;
    if ($("st-seed").value !== "") body.seed = Number($("st-seed").value);
    const { job, preview: p } = await call("/genome/render", { body });
    const card = el("div", { class: "item" },
      el("div", { class: "item-head" }, el("strong", {}, `${what} · ${state.voice.speaker}`), el("span", { class: "progress" }, "Queued…")),
      el("p", { class: "markup" }, p.markup),
      el("p", { class: "muted small" }, `Seed ${job.about.seed}` + (p.speed !== 1 ? ` · speed ${p.speed}` : "")));
    $("st-jobs").prepend(card);
    followJob(job.id, card);
  }

  async function render() {
    if (!needVoice()) return;
    if (!$("st-text").value.trim()) { toast("Type a line.", true); return; }
    const own = !$("st-blank").hidden ? $("st-blank-name").value.trim() : "";
    if (!$("st-blank").hidden && !own) { toast("Name your genome first.", true); $("st-blank-name").focus(); return; }
    if (own && !S.genes.length) { toast("Add at least one gene.", true); return; }
    const what = own || (S.picked && !S.changed ? (S.picked.name || `${S.picked.voice} #${S.picked.genome}`)
                                                : (S.picked ? "Edited genome" : "New genome"));
    try {
      // The plain line first, with the same seed, so the two differ only by the genome
      if ($("st-plain").checked && S.genes.length) await renderOne([], "Plain line");
      await renderOne(S.genes, S.genes.length ? what : "Plain line", own);
      if (own) { toast(`Saved “${own}” under My genomes.`); await load("private:" + own); }
    } catch (error) { toast(error.message, true); }
  }

  // -------------------------------------------------------------- wiring --
  $("st-find").addEventListener("input", () => S.lib && showList());
  $("st-source").addEventListener("change", () => S.lib && showList());
  $("st-pick").addEventListener("change", (e) => pick(e.target.value));
  $("st-text").addEventListener("input", schedulePreview);
  $("st-add").addEventListener("click", () => { S.genes.push({ feature: "emphasis", position: 0.5 }); changed(); });
  $("st-new").addEventListener("click", () => {
    S.picked = null; S.from = null; S.genes = []; S.changed = true;
    $("st-pick").value = ""; $("st-picked").textContent = "Empty: name it and add genes below."; $("st-remove").hidden = true;
    blank(true); $("st-blank-name").focus();
    renderGenes(); schedulePreview();
  });
  $("st-remove").addEventListener("click", remove);
  $("st-paste").addEventListener("click", paste);
  $("st-copy").addEventListener("click", () => {
    if (!S.genes.length) { toast("There are no genes to copy: this is the plain line.", true); return; }
    const from = S.picked ? (S.picked.name || `${S.picked.voice} genome ${S.picked.genome}`)
                          : S.from ? `${S.from}, edited in the studio` : "the genome studio";
    copyText(genomeText(S.genes, from), "The genome");
  });
  $("st-save").addEventListener("click", save);
  $("st-render").addEventListener("click", render);

  // From elsewhere on the page: open the studio on one genome
  function show(key) {
    S.pending = key;
    $("st-find").value = "";
    $("st-source").value = "";
    showTab("studio");          // calls open()
  }

  return { open, show, useGenes };
})();
