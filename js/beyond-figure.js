// panels for steps 12 and 13: pass@k coverage, process vs outcome reward, rl algorithms with a learnable
// length, and best-of-n against rl at equal kl. the math lives in beyond-model.js and rlhf-model.js.
(function () {
  "use strict";
  const B = window.Beyond;
  const R = window.RLHF;
  const W = window.ReasoningWalks;
  const C = window.MiniChart;
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const normal = W.gaussianFrom(Math.random);
  const $ = (sel, el = document) => el.querySelector(sel);
  const pct = (x) => (Number.isFinite(x) ? `${Math.round(100 * x)}%` : "–");
  const fmt = (x, d = 1) => (Number.isFinite(x) ? x.toFixed(d) : "–");
  const GROW_MS = 700;
  const FADE_MS = 1500;

  function palette(el) {
    const cs = getComputedStyle(el);
    const v = (n) => cs.getPropertyValue(n).trim();
    return { ink: v("--ink"), muted: v("--muted"), paper: v("--paper"), teal: v("--teal"), amber: v("--amber"), red: v("--red"), acceptFill: v("--accept-fill"), font: cs.fontFamily };
  }

  // ---------- a small canvas with walks ----------

  class Canvas {
    constructor(canvas, view, draw) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.view = view;
      this.draw = draw;
      this.trails = [];
      this.sampler = null;
      const [x0, x1, y0, y1] = view;
      canvas.style.aspectRatio = `${x1 - x0} / ${y1 - y0}`;
      new ResizeObserver(() => this.resize()).observe(canvas);
    }
    resize() {
      const dpr = window.devicePixelRatio || 1;
      const [x0, x1, y0, y1] = this.view;
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
    tick(now) {
      if (this.sampler && (!this.last || now - this.last > 120)) {
        this.trails.push({ ...this.sampler(), born: now });
        if (this.trails.length > 60) this.trails.shift();
        this.last = now;
      }
      this.trails = this.trails.filter((t) => now - t.born < GROW_MS + FADE_MS);
      this.render(now);
    }
    fillStatic() {
      this.trails = this.sampler ? Array.from({ length: 30 }, () => ({ ...this.sampler(), born: -1e9 })) : [];
    }
    circle(c, r, stroke, opts = {}) {
      const ctx = this.ctx;
      const [x, y] = this.toPx(c);
      ctx.beginPath();
      ctx.arc(x, y, r * this.scale, 0, 2 * Math.PI);
      if (opts.fill) {
        ctx.fillStyle = opts.fill;
        ctx.globalAlpha = opts.fillAlpha ?? 1;
        ctx.fill();
      }
      ctx.globalAlpha = opts.alpha ?? 1;
      ctx.setLineDash(opts.dash || []);
      ctx.strokeStyle = stroke;
      ctx.lineWidth = opts.width || 1;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }
    text(t, q, pal, opts = {}) {
      const [x, y] = this.toPx(q);
      this.ctx.font = `${opts.weight || 400} 12px ${pal.font}`;
      this.ctx.fillStyle = opts.color || pal.muted;
      this.ctx.textAlign = opts.align || "center";
      this.ctx.fillText(t, x + (opts.dx || 0), y + (opts.dy || 0));
    }
    render(now) {
      if (!this.width) return;
      const pal = palette(this.canvas);
      const ctx = this.ctx;
      const dpr = window.devicePixelRatio || 1;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, this.width, this.height);
      ctx.lineJoin = "round";
      this.draw(this, pal, "under");
      for (const t of this.trails) {
        const age = now - t.born;
        const last = t.points.length - 1;
        const shown = reducedMotion || t.born < 0 ? last + 1 : Math.min(last + 1, 1 + Math.floor((last * age) / GROW_MS));
        const fade = reducedMotion || t.born < 0 ? 0.7 : Math.max(0, 1 - Math.max(0, age - GROW_MS) / FADE_MS);
        const done = shown > last;
        ctx.globalAlpha = (t.hit && done ? 0.75 : 0.35) * fade;
        ctx.strokeStyle = t.hit && done ? pal.teal : pal.ink;
        ctx.lineWidth = t.hit && done ? 1.6 : 1;
        ctx.beginPath();
        for (let i = 0; i < shown; i++) {
          const [x, y] = this.toPx(t.points[i]);
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      this.draw(this, pal, "over");
    }
  }

  // a path with a constant offset, used by every canvas here
  function path(start, theta, sigmaEnd, steps) {
    const sd = sigmaEnd * Math.sqrt(W.DT);
    const pts = [start];
    for (let k = 0; k < steps; k++) {
      const d = W.driftAt(k);
      const p = pts[k];
      pts.push([p[0] + d[0] + theta[0] * W.DT + sd * normal(), p[1] + d[1] + theta[1] * W.DT + sd * normal()]);
    }
    return pts;
  }

  // ---------- step 11a: pass@k coverage ----------

  function mountCoverage(panel) {
    const stepInput = $("#cov-step", panel);
    const betaInput = $("#cov-beta", panel);
    const ks = Array.from({ length: 9 }, (_, i) => 2 ** i);
    const base = { theta: [0, 0], logSigma: Math.log(B.COVERAGE.baseSigma) };
    let history = [];
    let policy = base;
    const view = [-1.2, 1.0, -1.5, 1.45];
    const field = new Canvas($("#cov-field", panel), view, (f, pal, layer) => {
      if (layer !== "under") return;
      for (const t of B.COVERAGE.tasks) {
        const p = B.coverageHit(policy, t);
        f.circle(t.target, B.COVERAGE.radius, p > 0.05 ? pal.teal : pal.muted, { dash: p > 0.05 ? [] : [4, 4], fill: pal.teal, fillAlpha: Math.min(0.5, p), width: p > 0.05 ? 1.6 : 1 });
        f.text(pct(p), t.target, pal, { dx: B.COVERAGE.radius * f.scale + 6, dy: 4, align: "left" });
        const [sx, sy] = f.toPx(t.start);
        f.ctx.fillStyle = pal.ink;
        f.ctx.beginPath();
        f.ctx.arc(sx, sy, 3.5, 0, 2 * Math.PI);
        f.ctx.fill();
      }
    });
    field.sampler = () => {
      const t = B.COVERAGE.tasks[Math.floor(Math.random() * B.COVERAGE.tasks.length)];
      const pts = path(t.start, policy.theta, Math.exp(policy.logSigma), W.STEPS);
      const end = pts[W.STEPS];
      return { points: pts, hit: Math.hypot(end[0] - t.target[0], end[1] - t.target[1]) <= B.COVERAGE.radius };
    };
    const chart = C.lineChart($("#cov-chart", panel), {
      x: ks, xLog: true, yMax: 1, yFormat: pct, xFormat: String, xLabel: "k, attempts per task (log scale)",
      series: [{ name: "base model", color: "--amber", values: [] }, { name: "after RL", color: "--teal", values: [] }],
    });

    function retrain() {
      history = B.trainCoverage({ beta: +betaInput.value });
      $("#cov-beta-out", panel).textContent = betaInput.value;
      show();
    }
    function show() {
      const s = +stepInput.value;
      policy = s === 0 ? base : history[s - 1];
      $("#cov-step-out", panel).textContent = s;
      const b = B.passAtK(base, ks);
      const t = B.passAtK(policy, ks);
      chart.update({
        series: [{ name: "base model", color: "--amber", values: b }, { name: s ? `after ${s} RL steps` : "same as base", color: "--teal", values: t }],
        tooltip: (i) => `k = ${ks[i]}<br>base <b>${pct(b[i])}</b><br>after RL <b>${pct(t[i])}</b>`,
      });
      const solved = (p) => B.COVERAGE.tasks.filter((x) => B.coverageHit(p, x) >= 0.01).length;
      const cross = ks.find((k, i) => b[i] > t[i] && i > 0 && t[0] > b[0]);
      $("#cov-out", panel).innerHTML =
        `<div><span>pass@1, base → now</span><b>${pct(b[0])} → ${pct(t[0])}</b></div>` +
        `<div><span>pass@256, base → now</span><b>${pct(b[8])} → ${pct(t[8])}</b></div>` +
        `<div><span>Tasks solved at least 1% of the time</span><b>${solved(base)} → ${solved(policy)} of ${B.COVERAGE.tasks.length}</b></div>` +
        `<div><span>Spread of the policy</span><b>${fmt(Math.exp(policy.logSigma), 3)}</b></div>` +
        `<div><span>The base model wins from</span><b>${cross ? `k = ${cross}` : "never"}</b></div>`;
      field.trails = [];
      if (reducedMotion) field.fillStatic();
      field.render(performance.now());
    }
    stepInput.addEventListener("input", show);
    betaInput.addEventListener("input", retrain);
    retrain();
    return { tick: (now) => !reducedMotion && field.tick(now) };
  }

  // ---------- step 11b: process vs outcome reward ----------

  function mountProcess(panel) {
    const runs = { outcome: null, process: null };
    let snapshot = 0;
    let playing = false;
    let lastFrame = 0;
    const view = R.VIEW;
    const draw = (kind) => (f, pal, layer) => {
      if (layer !== "under") return;
      if (kind === "process") {
        // the corridor: dashed centre line and one ring per checkpoint
        f.ctx.setLineDash([3, 4]);
        f.ctx.strokeStyle = pal.muted;
        f.ctx.beginPath();
        for (let k = 0; k <= W.STEPS; k++) {
          const [x, y] = f.toPx(B.idealPoint(k));
          k ? f.ctx.lineTo(x, y) : f.ctx.moveTo(x, y);
        }
        f.ctx.stroke();
        f.ctx.setLineDash([]);
        for (const k of B.PROCESS.checkpoints) f.circle(B.idealPoint(k), B.corridor(k), pal.amber, { alpha: 0.8 });
        f.text("checkpoints", B.idealPoint(16), pal, { color: pal.amber, dy: -B.corridor(16) * f.scale - 6 });
      }
      f.circle(B.PROCESS.target, B.PROCESS.radius, pal.red, { fill: pal.acceptFill, fillAlpha: 0.8 });
      f.text("verifier", B.PROCESS.target, pal, { color: pal.red, dy: B.PROCESS.radius * f.scale + 14 });
      const [sx, sy] = f.toPx(B.PROCESS.start);
      f.ctx.fillStyle = pal.ink;
      f.ctx.beginPath();
      f.ctx.arc(sx, sy, 5, 0, 2 * Math.PI);
      f.ctx.fill();
      f.text("prompt x", B.PROCESS.start, pal, { dy: 20 });
    };
    const fields = ["outcome", "process"].map((kind) => {
      const c = new Canvas($(`#proc-field-${kind}`, panel), view, draw(kind));
      c.sampler = () => {
        const h = runs[kind] && runs[kind][snapshot];
        const theta = h ? h.theta : [0, 0];
        const sigma = h ? h.sigma : B.PROCESS.sigma;
        const pts = path(B.PROCESS.start, theta, sigma, W.STEPS);
        return { points: pts, hit: B.outcomeOf(pts) === 1 };
      };
      return c;
    });
    const chart = C.lineChart($("#proc-chart", panel), {
      x: [], yMax: 1, yFormat: pct, xLabel: "training step", empty: "Press Train both",
      series: [{ name: "outcome reward", color: "--amber", values: [] }, { name: "process reward", color: "--teal", values: [] }],
    });
    function show() {
      const o = runs.outcome;
      const p = runs.process;
      if (!o) return;
      const n = snapshot + 1;
      chart.update({
        x: o.slice(0, n).map((h) => h.step),
        series: [{ name: "outcome reward", color: "--amber", values: o.slice(0, n).map((h) => h.pass) },
                 { name: "process reward", color: "--teal", values: p.slice(0, n).map((h) => h.pass) }],
        tooltip: (i) => `step ${o[i].step}<br>outcome <b>${pct(o[i].pass)}</b><br>process <b>${pct(p[i].pass)}</b>`,
      });
      $("#proc-outcome", panel).textContent = pct(o[snapshot].pass);
      $("#proc-process", panel).textContent = pct(p[snapshot].pass);
      $("#proc-step", panel).textContent = o[snapshot].step;
    }
    $("[data-act=proc-train]", panel).addEventListener("click", () => {
      const seed = (Math.random() * 1e6) | 0;
      runs.outcome = B.trainProcess({ reward: "outcome", seed });
      runs.process = B.trainProcess({ reward: "process", seed });
      snapshot = 0;
      playing = true;
      fields.forEach((f) => (f.trails = []));
      show();
    });
    return {
      tick(now) {
        if (playing && now - lastFrame > 90) {
          lastFrame = now;
          snapshot = Math.min(runs.outcome.length - 1, snapshot + 1);
          if (snapshot === runs.outcome.length - 1) playing = false;
          show();
        }
        if (!reducedMotion) fields.forEach((f) => f.tick(now));
      },
    };
  }

  // ---------- step 12a: rl algorithms ----------

  const NAMES = { reinforce: "REINFORCE", rloo: "RLOO", grpo: "GRPO", drgrpo: "Dr. GRPO", ppo: "PPO-clip" };
  const SEEDS = [1, 2, 3, 4];

  function mountAlgorithms(panel) {
    const lrInput = $("#alg-lr", panel);
    const pick = $("#alg-pick", panel);
    const status = $("#alg-status", panel);
    let results = null;
    const view = [-1.12, 1.3, -1.45, 0.5];
    const field = new Canvas($("#alg-field", panel), view, (f, pal, layer) => {
      if (layer !== "under") return;
      f.circle(B.LENGTH.target, B.LENGTH.radius, pal.muted, { dash: [4, 4], fill: pal.teal, fillAlpha: 0.06 });
      const [sx, sy] = f.toPx(B.LENGTH.start);
      f.ctx.fillStyle = pal.ink;
      f.ctx.beginPath();
      f.ctx.arc(sx, sy, 5, 0, 2 * Math.PI);
      f.ctx.fill();
      f.text("prompt x", B.LENGTH.start, pal, { dy: 20 });
    });
    field.sampler = () => {
      const params = results ? results[pick.value].params : [0, 0, Math.log(B.LENGTH.sigma40), W.STEPS];
      const s = B.lengthSample(params, normal);
      // redraw the sample as a full path with the same length, so the canvas shows how long it ran
      const pts = path(B.LENGTH.start, [params[0], params[1]], Math.exp(params[2]), s.L);
      const end = pts[pts.length - 1];
      return { points: pts, hit: Math.hypot(end[0] - B.LENGTH.target[0], end[1] - B.LENGTH.target[1]) <= B.LENGTH.radius };
    };
    const success = C.lineChart($("#alg-success", panel), { x: [], yMax: 1, yFormat: pct, xLabel: "training step", series: [] });
    const gap = C.lineChart($("#alg-gap", panel), { x: [], yFormat: (v) => fmt(v, 0), xLabel: "training step", series: [] });
    const growth = C.lineChart($("#alg-length", panel), { x: [], yFromData: true, yFormat: (v) => fmt(v, 0), xLabel: "training step", series: [] });

    const smooth = (xs, w = 15) => xs.map((_, i) => {
      const win = xs.slice(Math.max(0, i - w), i + 1).filter(Number.isFinite);
      return win.length ? win.reduce((a, b) => a + b, 0) / win.length : NaN;
    });
    // the mean over seeds of one statistic at every step
    const across = (runs, key) => runs[0].history.map((_, i) => {
      const vals = runs.map((r) => r.history[i][key]).filter(Number.isFinite);
      return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : NaN;
    });

    function compute() {
      const lr = +lrInput.value;
      $("#alg-lr-out", panel).textContent = lr.toFixed(2);
      status.textContent = "Training 5 algorithms × 4 runs…";
      const out = {};
      const jobs = B.ALGORITHMS.flatMap((a) => SEEDS.map((seed) => ({ a, seed })));
      let i = 0;
      const next = () => {
        const { a, seed } = jobs[i++];
        (out[a] = out[a] || []).push(B.trainLength({ algorithm: a, seed, lr }));
        if (i < jobs.length) setTimeout(next, 0);
        else finish(out);
      };
      setTimeout(next, 20);
    }

    function finish(out) {
      results = {};
      const steps = out.grpo[0].history.map((h) => h.step);
      const rows = B.ALGORITHMS.map((a) => {
        const tail = (key) => {
          const v = across(out[a], key).slice(-50).filter(Number.isFinite);
          return v.reduce((x, y) => x + y, 0) / v.length;
        };
        // the walks on the canvas come from the first run's final policy
        results[a] = { params: out[a][0].params };
        return { a, success: tail("success"), correct: tail("lenCorrect"), wrong: tail("lenWrong") };
      });
      const g = (a) => smooth(across(out[a], "lenWrong").map((w, i) => w - across(out[a], "lenCorrect")[i]));
      success.update({
        x: steps,
        series: [{ name: "GRPO", color: "--amber", values: smooth(across(out.grpo, "success")) },
                 { name: "Dr. GRPO", color: "--teal", values: smooth(across(out.drgrpo, "success")) }],
        tooltip: (i) => `step ${steps[i]}`,
      });
      const gg = g("grpo");
      const gd = g("drgrpo");
      gap.update({
        x: steps,
        series: [{ name: "GRPO", color: "--amber", values: gg }, { name: "Dr. GRPO", color: "--teal", values: gd }],
        tooltip: (i) => `step ${steps[i]}<br>GRPO <b>${fmt(gg[i])}</b> steps<br>Dr. GRPO <b>${fmt(gd[i])}</b> steps`,
      });
      const lr0 = across(out.reinforce, "muL");
      const lp = across(out.ppo, "muL");
      growth.update({
        x: steps,
        series: [{ name: "REINFORCE", color: "--amber", values: lr0 }, { name: "PPO-clip", color: "--teal", values: lp }],
        tooltip: (i) => `step ${steps[i]}<br>REINFORCE <b>${fmt(lr0[i])}</b> steps<br>PPO-clip <b>${fmt(lp[i])}</b> steps`,
      });
      $("#alg-table", panel).innerHTML =
        `<thead><tr><th>Algorithm</th><th>Success</th><th>Right answers, mean length</th><th>Wrong answers, mean length</th><th>Wrong − right</th></tr></thead><tbody>` +
        rows.map((r) => `<tr><th>${NAMES[r.a]}</th><td>${pct(r.success)}</td><td>${fmt(r.correct)}</td><td>${fmt(r.wrong)}</td><td>${fmt(r.wrong - r.correct)}</td></tr>`).join("") +
        "</tbody>";
      status.textContent = `Mean of ${SEEDS.length} runs per algorithm, last 50 of 300 steps, learning rate ${(+lrInput.value).toFixed(2)}.`;
      field.trails = [];
    }

    lrInput.addEventListener("change", compute);
    lrInput.addEventListener("input", () => ($("#alg-lr-out", panel).textContent = (+lrInput.value).toFixed(2)));
    pick.addEventListener("change", () => (field.trails = []));
    let started = false;
    return {
      onShow() {
        if (!started) {
          started = true;
          compute();
        }
      },
      tick: (now) => !reducedMotion && field.tick(now),
    };
  }

  // ---------- step 12b: best-of-n against rl at equal kl ----------

  function mountBestOfN(panel) {
    const status = $("#bon-status", panel);
    const chart = C.lineChart($("#bon-chart", panel), {
      x: [], markers: true, xLabel: "√KL to base model", xFormat: (v) => fmt(v, 1), yFormat: (v) => fmt(v, 1), empty: "Computing…", series: [],
    });
    let done = false;
    function compute() {
      if (done) return;
      done = true;
      setTimeout(() => {
        const model = R.fitRewardModel(R.collectPairs(200, 1));
        const base = R.evaluate(R.refPolicy(), model, 4000);
        const rl = R.betaSweep(model).slice(0, 8).map((s) => ({ x: Math.sqrt(s.kl), proxy: s.proxy - base.proxy, util: s.utility - base.utility, label: `RL, β = ${s.beta}` }));
        const uniform = R.seededUniform(5);
        const g = () => Math.sqrt(-2 * Math.log(uniform() + 1e-12)) * Math.cos(2 * Math.PI * uniform());
        const ref = R.refPolicy();
        const bon = [1, 2, 4, 16, 64, 256, 1024, 4096].map((N) => {
          const trials = N >= 1024 ? 250 : 1200;
          let px = 0;
          let u = 0;
          for (let t = 0; t < trials; t++) {
            let best = null;
            let br = -Infinity;
            for (let i = 0; i < N; i++) {
              const y = R.sampleEnd(ref, g);
              const r = R.rmReward(model, y);
              if (r > br) [br, best] = [r, y];
            }
            px += br;
            u += R.trueUtility(best);
          }
          // exact here because answers are continuous; for real models it is only an upper bound
          const kl = Math.log(N) - (N - 1) / N;
          return { x: Math.sqrt(kl), proxy: px / trials - base.proxy, util: u / trials - base.utility, label: `best-of-${N}` };
        });
        // one x grid holding both families, with gaps where a family has no point
        const all = [...rl.map((p) => ({ ...p, fam: "rl" })), ...bon.map((p) => ({ ...p, fam: "bon" }))].sort((a, b) => a.x - b.x);
        const xs = all.map((p) => p.x);
        const join = (fam, key) => {
          const pts = all.map((p, i) => [i, p]).filter(([, p]) => p.fam === fam);
          const out = new Array(all.length).fill(NaN);
          pts.forEach(([i, p], j) => {
            out[i] = p[key];
            // fill the gap to the next point of the same family, so the line is continuous
            if (j < pts.length - 1) {
              const [i2, p2] = pts[j + 1];
              for (let m = i + 1; m < i2; m++) out[m] = p[key] + ((p2[key] - p[key]) * (xs[m] - p.x)) / (p2.x - p.x);
            }
          });
          return out;
        };
        chart.update({
          x: xs,
          series: [
            { name: "true utility, RL", color: "--teal", values: join("rl", "util") },
            { name: "true utility, best-of-N", color: "--teal", dash: [6, 4], values: join("bon", "util") },
            { name: "proxy reward, RL", color: "--amber", values: join("rl", "proxy") },
            { name: "proxy reward, best-of-N", color: "--amber", dash: [6, 4], values: join("bon", "proxy") },
          ],
          tooltip: (i) => {
            const p = all[i];
            return `${p.label}<br>√KL ${fmt(p.x, 2)}<br>true gain <b>${fmt(p.util, 2)}</b><br>proxy gain <b>${fmt(p.proxy, 2)}</b>`;
          },
          markers: false,
        });
        const peak = (arr) => arr.reduce((a, b) => (b.util > a.util ? b : a));
        const pr = peak(rl);
        const pb = peak(bon);
        status.innerHTML = `Best true gain: <b>RL ${fmt(pr.util, 2)}</b> (${pr.label}, √KL ${fmt(pr.x, 2)}) against <b>best-of-N ${fmt(pb.util, 2)}</b> (${pb.label}, √KL ${fmt(pb.x, 2)}). Both overoptimize past their peak.`;
      }, 30);
    }
    return { onShow: compute, tick: () => {} };
  }

  const panels = {};
  const byId = (id) => document.getElementById(id);
  const cov = mountCoverage(byId("beyond-limits"));
  const proc = mountProcess(byId("beyond-limits"));
  panels["beyond-limits"] = { tick: (now) => { cov.tick(now); proc.tick(now); } };
  const alg = mountAlgorithms(byId("beyond-algorithms"));
  const bon = mountBestOfN(byId("beyond-algorithms"));
  panels["beyond-algorithms"] = { onShow: () => { alg.onShow(); bon.onShow(); }, tick: (now) => alg.tick(now) };
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
