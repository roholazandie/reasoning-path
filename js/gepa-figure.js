// panels for steps 9 and 10: one gepa iteration at a time, then the candidate pool and gepa vs grpo.
// the math lives in gepa-model.js; this file only draws and wires controls.
(function () {
  "use strict";
  const G = window.GEPA;
  const C = window.MiniChart;
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const normal = G.gaussianFrom(Math.random);
  const $ = (sel, el = document) => el.querySelector(sel);
  const pct = (x) => `${Math.round(100 * x)}%`;
  const esc = (t) => t.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
  const BEST = G.testSuccess((t) => [G.NEEDED.H.shift + G.NEEDED.H.tilt * t.height, G.NEEDED.V.shift + G.NEEDED.V.tilt * t.height]);
  const GROW_MS = 600;
  const FADE_MS = 1600;

  function palette(el) {
    const cs = getComputedStyle(el);
    const v = (n) => cs.getPropertyValue(n).trim();
    return { ink: v("--ink"), muted: v("--muted"), paper: v("--paper"), teal: v("--teal"), amber: v("--amber"), font: cs.fontFamily };
  }

  // ---------- task field ----------

  class TaskField {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.trails = [];
      this.marks = [];
      this.highlight = new Set();
      this.offsetOf = () => [0, 0];
      const [x0, x1, y0, y1] = G.VIEW;
      canvas.style.aspectRatio = `${x1 - x0} / ${y1 - y0}`;
      new ResizeObserver(() => this.resize()).observe(canvas);
    }
    resize() {
      const dpr = window.devicePixelRatio || 1;
      const [x0, x1, y0, y1] = G.VIEW;
      this.width = this.canvas.clientWidth;
      this.height = this.canvas.clientHeight;
      if (!this.width) return;
      this.canvas.width = Math.round(this.width * dpr);
      this.canvas.height = Math.round(this.height * dpr);
      this.scale = Math.min(this.width / (x1 - x0), this.height / (y1 - y0));
      this.ox = this.width / 2 - (this.scale * (x0 + x1)) / 2;
      this.oy = this.height / 2 + (this.scale * (y0 + y1)) / 2;
      this.render(performance.now());
    }
    toPx(q) {
      return [this.ox + q[0] * this.scale, this.oy - q[1] * this.scale];
    }
    spawn(now) {
      const task = G.TASKS[Math.floor(Math.random() * G.TASKS.length)];
      this.trails.push({ points: G.walkPoints(this.offsetOf(task), task, normal), born: now });
      if (this.trails.length > 60) this.trails.shift();
    }
    tick(now) {
      if (!this.last || now - this.last > 110) {
        this.spawn(now);
        this.last = now;
      }
      this.trails = this.trails.filter((t) => now - t.born < GROW_MS + FADE_MS);
      this.render(now);
    }
    render(now) {
      if (!this.width) return;
      const pal = palette(this.canvas);
      const ctx = this.ctx;
      const dpr = window.devicePixelRatio || 1;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, this.width, this.height);
      ctx.font = `12px ${pal.font}`;
      for (const task of G.TASKS) {
        const [tx, ty] = this.toPx(task.target);
        const on = this.highlight.has(task.id);
        ctx.beginPath();
        ctx.arc(tx, ty, G.RADIUS * this.scale, 0, 2 * Math.PI);
        ctx.setLineDash(on ? [] : [4, 4]);
        ctx.strokeStyle = on ? pal.amber : pal.muted;
        ctx.lineWidth = on ? 2 : 1;
        ctx.globalAlpha = on ? 1 : 0.7;
        ctx.stroke();
        ctx.setLineDash([]);
        const [sx, sy] = this.toPx(task.start);
        ctx.globalAlpha = 1;
        ctx.fillStyle = pal.ink;
        ctx.beginPath();
        ctx.arc(sx, sy, 4, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillStyle = pal.muted;
        ctx.textAlign = "right";
        ctx.fillText(`task ${task.id + 1}`, sx - 8, sy + 4);
      }
      ctx.lineWidth = 1;
      for (const t of this.trails) {
        const age = now - t.born;
        const shown = reducedMotion ? t.points.length : Math.min(t.points.length, 1 + Math.floor(((t.points.length - 1) * age) / GROW_MS));
        const fade = reducedMotion ? 0.6 : Math.max(0, 1 - Math.max(0, age - GROW_MS) / FADE_MS);
        ctx.globalAlpha = 0.35 * fade;
        ctx.strokeStyle = pal.ink;
        ctx.beginPath();
        for (let i = 0; i < shown; i++) {
          const [x, y] = this.toPx(t.points[i]);
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.stroke();
      }
      // minibatch endpoints: hollow for the parent, filled for the child
      for (const m of this.marks) {
        const [x, y] = this.toPx(m.end);
        ctx.globalAlpha = 1;
        ctx.beginPath();
        ctx.arc(x, y, 4.5, 0, 2 * Math.PI);
        if (m.child) {
          ctx.fillStyle = pal.teal;
          ctx.fill();
        } else {
          ctx.fillStyle = pal.paper;
          ctx.fill();
          ctx.strokeStyle = pal.ink;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;
    }
  }

  // ---------- prompt rendering ----------

  function promptHtml(cand, changed) {
    return G.MODULES.map((name) => {
      const mod = cand.modules[name];
      const lines = [`<li class="base">${esc(G.BASE_TEXT[name])}</li>`]
        .concat(mod.lines.map((l) => {
          const fresh = changed && changed.has(l.id);
          return `<li class="${fresh ? "fresh" : ""}">${esc(l.text)} <span class="tag">${l.m > 0 ? "+" : ""}${l.m.toFixed(2)}</span></li>`;
        }));
      const title = name === "H" ? "Module H: horizontal aim" : "Module V: vertical aim";
      return `<div class="module"><div class="module-name">${title}</div><ol>${lines.join("")}</ol></div>`;
    }).join("");
  }

  // ---------- shared settings, set by step 9's controls ----------

  const settings = { quality: G.Q0, feedback: "text", minibatch: 3, paretoSamples: 4 };
  const settingListeners = [];
  function readSettings(panel) {
    settings.quality = +$("#g9-quality", panel).value;
    settings.feedback = $("#g9-text", panel).checked ? "text" : "binary";
    settings.minibatch = +$("#g9-minibatch", panel).value;
    settings.paretoSamples = +$("#g9-pareto", panel).value;
    $("#g9-quality-out", panel).textContent = settings.quality.toFixed(2);
    $("#g9-minibatch-out", panel).textContent = settings.minibatch;
    $("#g9-pareto-out", panel).textContent = settings.paretoSamples;
    settingListeners.forEach((f) => f());
  }
  const isDefault = () => settings.quality === G.Q0 && settings.feedback === "text" && settings.minibatch === 3 && settings.paretoSamples === 4;

  // ---------- step 9: one iteration at a time ----------

  function mountStep9(panel) {
    const field = new TaskField($("#gepa-field", panel));
    let gepa = null;
    let current = null;
    let log = [];

    function reset() {
      gepa = G.createGepa({ seed: 11, merge: false, ...settings });
      current = gepa.best();
      log = [];
      field.marks = [];
      field.highlight = new Set();
      field.offsetOf = (t) => G.promptOffset(current, t);
      field.trails = [];
      $("#g9-event", panel).innerHTML = `<p class="muted">Press <b>Run one iteration</b>. GEPA will pick a candidate, try it on ${settings.minibatch} task${settings.minibatch > 1 ? "s" : ""}, read the feedback, and rewrite one module.</p>`;
      show(null);
    }

    function show(event) {
      $("#g9-prompt", panel).innerHTML = promptHtml(current, event && event.accepted ? new Set(event.child.modules[event.module].lines.map((l) => l.id)) : null);
      $("#g9-prompt-title", panel).textContent = `Prompt of candidate #${current.id}${current.id === gepa.best().id ? ", the best so far" : ""}`;
      $("#g9-rollouts", panel).textContent = gepa.rollouts;
      $("#g9-pool", panel).textContent = gepa.pool.length;
      $("#g9-test", panel).textContent = pct(G.testSuccess((t) => G.promptOffset(gepa.best(), t)));
      $("#g9-log", panel).innerHTML = log.slice(-8).reverse().map((e) =>
        `<li><span class="${e.accepted ? "ok" : "no"}">${e.accepted ? "kept" : "dropped"}</span> #${e.parent} → module ${e.module}: ${e.scoreBefore}/${e.batch.length} → ${e.scoreAfter}/${e.batch.length}</li>`).join("");
      if (reducedMotion) {
        field.trails = [];
        for (let i = 0; i < 24; i++) field.spawn(-1e9);
      }
      field.render(performance.now());
    }

    function describe(e) {
      const rows = e.batch.map((id, i) => {
        const p = e.parentTraces[i];
        const c = e.childTraces[i];
        const text = (r) => (settings.feedback === "binary" ? (r.hit ? "Pass." : "Fail.") : G.feedbackText(r));
        return `<tr><th>Task ${id + 1}</th><td>${esc(text(p))}</td><td>${esc(text(c))}</td></tr>`;
      }).join("");
      const b = e.batch.length;
      const verdict = e.accepted
        ? `The child scored <b>${e.scoreAfter}/${b}</b> against the parent's <b>${e.scoreBefore}/${b}</b> on the same tasks, so it joins the pool and is scored on all 8 tasks (${8 * settings.paretoSamples} more rollouts).`
        : `The child scored <b>${e.scoreAfter}/${b}</b> against the parent's <b>${e.scoreBefore}/${b}</b>. That is no improvement, so it is dropped. Only the ${2 * b} minibatch rollouts were spent.`;
      const how = settings.feedback === "binary"
        ? "With text feedback off, it sees only pass or fail, so it guesses a direction."
        : `The reflector writes a ${e.kind === "tilt" ? "height-dependent" : "shared"} correction to module ${e.module}.`;
      return `
        <ol class="phases">
          <li><b>Select.</b> Candidate #${e.parent} from the Pareto front. Module ${e.module} is next in turn.</li>
          <li><b>Run on a minibatch.</b> Task${b > 1 ? "s" : ""} ${e.batch.map((i) => i + 1).join(", ")}, one rollout each.</li>
          <li><b>Read the feedback</b> and <b>reflect.</b> ${how}</li>
          <li><b>Try the child</b> on the same tasks, then keep or drop it.</li>
        </ol>
        <div class="table-wrap tight"><table><thead><tr><th></th><th>Parent feedback</th><th>Child feedback</th></tr></thead><tbody>${rows}</tbody></table></div>
        <p>${verdict}</p>`;
    }

    function step() {
      let e = gepa.iterate();
      while (e.type !== "mutate") e = gepa.iterate();
      log.push(e);
      current = e.accepted ? e.child : gepa.byId(e.parent);
      field.offsetOf = (t) => G.promptOffset(current, t);
      field.trails = [];
      field.highlight = new Set(e.batch);
      field.marks = e.batch.flatMap((id, i) => [{ end: e.parentTraces[i].end }, { end: e.childTraces[i].end, child: true }]);
      $("#g9-event", panel).innerHTML = describe(e);
      show(e);
    }

    panel.addEventListener("click", (ev) => {
      const act = ev.target.closest("[data-act]")?.dataset.act;
      if (act === "g9-step") step();
      if (act === "g9-ten") for (let i = 0; i < 10; i++) step();
      if (act === "g9-reset") reset();
    });
    ["#g9-quality", "#g9-minibatch", "#g9-pareto", "#g9-text"].forEach((id) =>
      $(id, panel).addEventListener("input", () => {
        readSettings(panel);
        reset();
      }));
    reset();
    return { tick: (now) => { if (!reducedMotion) field.tick(now); } };
  }

  // ---------- step 10: pareto pool and gepa vs grpo ----------

  const GEPA_BUDGET = 6000;
  const GRPO_BUDGET = 24000;
  const SEEDS = [1, 2, 3, 4, 5, 6];
  const GRID = Array.from({ length: 44 }, (_, i) => Math.round(30 * (GRPO_BUDGET / 30) ** (i / 43)));

  // the mean over seeds of each run's best test success at or before every grid point
  function meanCurve(histories, budget) {
    return GRID.map((x) => {
      if (x > budget) return NaN;
      const vals = histories.map((h) => {
        let v = h[0].test;
        for (const p of h) if (p.rollouts <= x) v = p.test;
        return v;
      });
      return vals.reduce((a, b) => a + b, 0) / vals.length;
    });
  }

  function mountStep10(panel) {
    const status = $("#g10-status", panel);
    const fmtK = (x) => (x >= 1000 ? `${x / 1000}k` : String(x));
    const base = { x: GRID, xLog: true, yMax: 1, yFormat: pct, xFormat: fmtK, xLabel: "rollouts (log scale)", empty: "Computing…" };
    const tip = (names, curves) => (i) => `${GRID[i]} rollouts<br>` + names.map((n, k) => `${n} <b>${Number.isFinite(curves[k][i]) ? pct(curves[k][i]) : "–"}</b>`).join("<br>");
    const headline = C.lineChart($("#g10-chart", panel), { ...base, x: [], series: [] });
    const minis = ["merge", "greedy", "random"].map((k) => C.lineChart($(`#g10-${k}`, panel), { ...base, x: [], series: [] }));
    let computedFor = null;

    // the pool table shows one run, so the reader can watch selection and merge happen
    let gepa = null;
    function resetPool() {
      gepa = G.createGepa({ seed: 3, merge: true, ...settings });
      renderPool();
    }
    function renderPool() {
      const front = gepa.front();
      const rows = gepa.pool.slice(-14).map((c) => {
        const s = gepa.scores.get(c.id);
        const bestPer = G.TASKS.map((_, t) => Math.max(...gepa.pool.map((d) => gepa.scores.get(d.id)[t])));
        const cells = s.map((v, t) => `<td class="cell${v === bestPer[t] ? " top" : ""}" style="--v:${v}" title="task ${t + 1}: ${pct(v)}">${Math.round(v * settings.paretoSamples)}</td>`).join("");
        const origin = c.origin === "merge" ? `merge of #${c.parents[0]} and #${c.parents[1]}` : c.origin === "base" ? "base prompt" : `from #${c.parent}`;
        const f = front.get(c.id);
        return `<tr><th>#${c.id}</th><td class="origin">${origin}</td>${cells}<td class="freq">${f ? `best on ${f}` : ""}</td></tr>`;
      }).join("");
      $("#g10-pool", panel).innerHTML = `<thead><tr><th>Candidate</th><th>Made by</th>${G.TASKS.map((t) => `<th>T${t.id + 1}</th>`).join("")}<th>On the front</th></tr></thead><tbody>${rows}</tbody>`;
      $("#g10-pool-rollouts", panel).textContent = gepa.rollouts;
      $("#g10-pool-best", panel).textContent = pct(G.testSuccess((t) => G.promptOffset(gepa.best(), t)));
    }
    panel.addEventListener("click", (ev) => {
      const act = ev.target.closest("[data-act]")?.dataset.act;
      if (act === "g10-run") {
        const until = gepa.rollouts + 300;
        while (gepa.rollouts < until) gepa.iterate();
        renderPool();
      }
      if (act === "g10-reset") resetPool();
      if (act === "g10-rerun") compute(true);
    });
    settingListeners.push(() => {
      resetPool();
      $("#g10-settings", panel).textContent = isDefault() ? "default settings" : "step 9's changed settings";
    });

    function compute(force) {
      const key = JSON.stringify(settings);
      if (!force && computedFor !== null) return;
      computedFor = key;
      const used = { ...settings };
      const jobs = [
        ["gepa", { merge: true }], ["nomerge", {}], ["greedy", { selection: "greedy", merge: true }], ["random", { reflective: false, merge: true }],
      ].flatMap(([k, opt]) => SEEDS.map((seed) => ({ key: k, run: () => G.runGepa({ ...used, ...opt, seed }, GEPA_BUDGET).history })))
        .concat(SEEDS.map((seed) => ({ key: "grpo", run: () => G.runGrpo({ seed, budget: GRPO_BUDGET }).history })));
      const out = {};
      let done = 0;
      // one run per timeout keeps the page responsive while 30 runs compute
      const next = () => {
        const job = jobs[done++];
        (out[job.key] = out[job.key] || []).push(job.run());
        status.textContent = `Running ${done} of ${jobs.length} runs…`;
        if (done < jobs.length) setTimeout(next, 0);
        else finish(out);
      };
      setTimeout(next, 30);
    }

    function finish(out) {
      const curve = (k) => meanCurve(out[k], k === "grpo" ? GRPO_BUDGET : GEPA_BUDGET);
      const g = curve("gepa");
      const r = curve("grpo");
      headline.update({ x: GRID, series: [{ name: "GEPA, prompt only", color: "--teal", values: g }, { name: "GRPO, weights", color: "--amber", values: r }], tooltip: tip(["GEPA", "GRPO"], [g, r]) });
      const labels = { merge: ["without merge", "nomerge"], greedy: ["greedy selection", "greedy"], random: ["random edits", "random"] };
      ["merge", "greedy", "random"].forEach((k, i) => {
        const alt = curve(labels[k][1]);
        minis[i].update({ x: GRID, series: [{ name: "full GEPA", color: "--teal", values: g }, { name: labels[k][0], color: "--amber", values: alt }], tooltip: tip(["full GEPA", labels[k][0]], [g, alt]) });
      });
      const level = 0.85 * BEST;
      const med = (k) => {
        const a = out[k].map((h) => G.rolloutsToReach(h, level)).sort((x, y) => x - y);
        return a[Math.floor(a.length / 2)];
      };
      const fmt = (x) => (Number.isFinite(x) ? x.toLocaleString("en-US") : "more than the budget");
      const which = computedFor === JSON.stringify({ quality: G.Q0, feedback: "text", minibatch: 3, paretoSamples: 4 })
        ? "default settings"
        : `step 9's settings (quality ${settings.quality.toFixed(2)}, text feedback ${settings.feedback === "text" ? "on" : "off"}, minibatch ${settings.minibatch}, ${settings.paretoSamples} samples per task)`;
      status.innerHTML = `With ${which}. Median rollouts to reach ${pct(level)} success (85% of the best possible), over ${SEEDS.length} runs each: ` +
        `<b>GEPA ${fmt(med("gepa"))}</b>, <b>GRPO ${fmt(med("grpo"))}</b>, without merge ${fmt(med("nomerge"))}, greedy ${fmt(med("greedy"))}, random edits ${fmt(med("random"))}.`;
    }

    resetPool();
    return { onShow: compute, tick: () => {} };
  }

  const panels = {};
  const byId = (id) => document.getElementById(id);
  panels["gepa-loop"] = mountStep9(byId("gepa-loop"));
  panels["gepa-search"] = mountStep10(byId("gepa-search"));
  window.addEventListener("panel:show", (e) => {
    const p = panels[e.detail];
    if (p && p.onShow) p.onShow();
  });
  function loop(now) {
    for (const [id, p] of Object.entries(panels)) if (!byId(id).hidden) p.tick(now);
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
})();
