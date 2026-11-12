// tab shell and the best-of-k chart in step 1.
(function () {
  "use strict";
  const tabs = [...document.querySelectorAll("[role=tab]")];
  const panelOf = (tab) => document.getElementById(tab.getAttribute("aria-controls"));

  function select(tab, focus) {
    tabs.forEach((t) => {
      const on = t === tab;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      panelOf(t).hidden = !on;
    });
    if (focus) tab.focus();
    const id = tab.getAttribute("aria-controls");
    history.replaceState(null, "", `#${id}`);
    document.querySelector("main").scrollTop = 0;
    window.scrollTo(0, 0);
    window.dispatchEvent(new CustomEvent("panel:show", { detail: id }));
  }

  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => select(tab));
    tab.addEventListener("keydown", (e) => {
      const step = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 }[e.key];
      if (step) {
        e.preventDefault();
        select(tabs[(i + step + tabs.length) % tabs.length], true);
      }
      if (e.key === "Home") select(tabs[0], true);
      if (e.key === "End") select(tabs[tabs.length - 1], true);
    });
  });
  // in-text links like href="#rl" switch tabs instead of jumping
  document.addEventListener("click", (e) => {
    const link = e.target.closest("a[href^='#']");
    const tab = link && tabs.find((t) => `#${t.getAttribute("aria-controls")}` === link.getAttribute("href"));
    if (!tab) return;
    e.preventDefault();
    select(tab);
  });
  document.querySelectorAll("[data-next]").forEach((btn) =>
    btn.addEventListener("click", () => {
      const i = tabs.findIndex((t) => t.getAttribute("aria-selected") === "true");
      select(tabs[Math.min(tabs.length - 1, i + 1)]);
    }),
  );

  const start = tabs.find((t) => `#${t.getAttribute("aria-controls")}` === location.hash) || tabs[0];
  select(start);

  // best-of-k and chain length use the same walks as the figure in step 1, and follow its sliders
  const M = window.ReasoningWalks;
  const fig = document.querySelector("figure[data-variant=walks]");
  let variantNow = { ...M.VARIANTS.walks };
  const pOf = (v) => M.successRate(v, v.prompts[0], M.zeroWeights(v), true);
  const ks = Array.from({ length: 32 }, (_, i) => i + 1);
  const lengths = Array.from({ length: 31 }, (_, i) => 20 + 2 * i);
  const slider = document.getElementById("bok-k");
  const pct = (x) => `${Math.round(100 * x)}%`;
  let p = 0;
  let curve = [];
  let lengthCurve = [];
  let majority = [];
  const chart = window.MiniChart.lineChart(document.getElementById("bok-chart"), {
    x: ks, series: [],
    xLabel: "k, number of sampled paths", yMax: 1, yFormat: pct,
    tooltip: (i) => `k = ${ks[i]}<br>best-of-k <b>${pct(curve[i])}</b><br>majority vote <b>${pct(majority[i])}</b>`,
  });
  const lengthChart = window.MiniChart.lineChart(document.getElementById("len-chart"), {
    x: lengths, series: [{ name: "success of one path", color: "--teal", values: [] }],
    xLabel: "chain length, steps", yMax: 1, yFormat: pct,
    tooltip: (i) => `${lengths[i]} steps<br>one path lands <b>${pct(lengthCurve[i])}</b>`,
  });
  function recompute() {
    p = pOf(variantNow);
    curve = ks.map((k) => 1 - (1 - p) ** k);
    majority = M.majorityRates(variantNow, variantNow.prompts[0], M.zeroWeights(variantNow), ks, 1200);
    lengthCurve = lengths.map((n) => pOf({ ...variantNow, steps: n }));
    chart.update({ series: [
      { name: "best-of-k, with a verifier", color: "--teal", values: curve },
      { name: "majority vote, no verifier", color: "--amber", values: majority },
    ] });
    lengthChart.update({
      series: [{ name: "success of one path", color: "--teal", values: lengthCurve }],
      highlight: lengths.indexOf(M.stepsOf(variantNow)),
    });
    show();
  }
  function show() {
    const k = +slider.value;
    document.getElementById("bok-p").textContent = pct(p);
    document.getElementById("bok-k-out").textContent = k;
    document.getElementById("bok-out").textContent = pct(curve[k - 1]);
    document.getElementById("vote-out").textContent = pct(majority[k - 1]);
    chart.update({ highlight: k - 1 });
  }
  slider.addEventListener("input", show);
  fig.addEventListener("walks:change", (e) => {
    variantNow = e.detail.variant;
    recompute();
  });
  recompute();
})();
