// renders.js | The Renders tab: every line spoken or rendered on the host, by anyone, to listen to.
"use strict";

const Renders = (() => {
  const R = { page: 1, pages: 1 };

  function item(r) {
    const title = r.genome || (r.genes ? `Edited genome (${r.genes} gene${r.genes === 1 ? "" : "s"})` : "Plain line");
    const details = [`by ${r.by || "?"}`, r.made, r.seed !== null && r.seed !== undefined ? `seed ${r.seed}` : null,
                     r.speed && r.speed !== 1 ? `speed ${r.speed}` : null].filter(Boolean).join(" · ");
    return el("li", { class: "item" },
      el("div", { class: "item-head" }, el("strong", {}, `${title} · ${r.voice || ""}`), el("span", { class: "muted small" }, details)),
      el("p", {}, r.text || ""),
      r.genes ? el("p", { class: "markup" }, r.markup || "") : null,
      el("audio", { controls: true, preload: "none", src: `${API}/renders/${encodeURIComponent(r.id)}/audio` }),
      r.mine ? stars(r) : null);
  }

  // Your own renders: a score from 1 to 5, kept apart from the listening piles' MOS
  function stars(r) {
    const box = el("div", { class: "row own-score" }, el("span", { class: "muted small" }, "Your rating:"));
    const draw = () => box.querySelectorAll("button").forEach((b) => {
      b.classList.toggle("on", Number(b.dataset.n) <= (r.my_score || 0));
      b.setAttribute("aria-pressed", String(Number(b.dataset.n) === r.my_score));
    });
    for (let n = 1; n <= 5; n++) {
      box.append(el("button", { type: "button", class: "quiet star", "data-n": n, "aria-label": `${n} of 5`, onclick: async () => {
        const score = r.my_score === n ? null : n;            // clicking your score again clears it
        try {
          await call(`/renders/${encodeURIComponent(r.id)}/rating`, { body: { score } });
          r.my_score = score; draw();
        } catch (error) { toast(error.message, true); }
      } }, "★"));
    }
    draw();
    return box;
  }

  async function load(page = R.page) {
    const params = new URLSearchParams({ page, q: $("rn-q").value.trim() });
    if ($("rn-mine").checked) params.set("mine", "1");
    $("rn-count").textContent = "Loading…";
    try {
      const data = await call("/renders?" + params);
      R.page = data.page;
      R.pages = data.pages;
      $("rn-list").replaceChildren(...data.renders.map(item));
      $("rn-count").textContent = data.total
        ? `${data.total} render${data.total === 1 ? "" : "s"}` : "No renders yet: make one in Speak or the Genome studio.";
      $("rn-page").textContent = `${R.page} of ${R.pages}`;
      $("rn-prev").disabled = R.page <= 1;
      $("rn-next").disabled = R.page >= R.pages;
    } catch (error) { $("rn-count").textContent = error.message; }
  }

  // Only one clip plays at a time
  $("rn-list").addEventListener("play", (e) => {
    document.querySelectorAll("#rn-list audio").forEach((a) => { if (a !== e.target) a.pause(); });
  }, true);
  $("rn-search").addEventListener("submit", (e) => { e.preventDefault(); load(1); });
  $("rn-mine").addEventListener("change", () => load(1));
  $("rn-prev").addEventListener("click", () => load(R.page - 1));
  $("rn-next").addEventListener("click", () => load(R.page + 1));

  return { open: () => load(1) };
})();
