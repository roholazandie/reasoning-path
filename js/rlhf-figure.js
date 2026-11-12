// interactive panels for part 2 (rlhf): preferences, reward model, rl with kl, dpo, and rlvr vs rlhf.
// the math lives in rlhf-model.js; this file only draws and wires controls.
(function () {
  "use strict";
  const R = window.RLHF;
  const W = window.ReasoningWalks;
  const C = window.MiniChart;
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const normal = W.gaussianFrom(Math.random);
  const HELD_OUT = R.collectPairs(500, 2);
  const TRAIN_STEPS = 300;
  const MIN_PAIRS = 10;
  const GROW_MS = 600;
  const FADE_MS = 1400;
  const SPAWN_MS = 90;
  const fmt = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : "–");
  const pct = (x) => (Number.isFinite(x) ? `${Math.round(100 * x)}%` : "–");
  const $ = (sel, el = document) => el.querySelector(sel);

  // ---------- shared label store ----------

  const store = { pairs: [], model: R.fitRewardModel([]), version: 0, listeners: [] };
  function setPairs(pairs) {
    store.pairs = pairs;
    store.model = R.fitRewardModel(pairs);
    store.version++;
    store.listeners.forEach((f) => f());
  }
  // how strictly simulated labelers follow the hidden utility; step 4's slider sets it for new labels
  const labeler = { noise: 1 };
  function autoPairs(n, policy = R.refPolicy()) {
    return Array.from({ length: n }, () => {
      const a = R.sampleEnd(policy, normal);
      const b = R.sampleEnd(policy, normal);
      return { a, b, winner: R.simulatedLabel(a, b, Math.random, 0, labeler.noise), source: "auto" };
    });
  }
  // an ai labeler is consistent (low noise) but has a taste of its own: it also likes answers further up
  const AI_LABELER = { bias: 1.5, noise: 0.5, axis: 1 };
  function aiPairs(n) {
    const ref = R.refPolicy();
    return Array.from({ length: n }, () => {
      const a = R.sampleEnd(ref, normal);
      const b = R.sampleEnd(ref, normal);
      return { a, b, winner: R.simulatedLabel(a, b, Math.random, AI_LABELER.bias, AI_LABELER.noise, AI_LABELER.axis), source: "ai" };
    });
  }
  const proxyReward = (y) => R.rmReward(store.model, y);
  const storeHeat = () => ({ key: `store${store.version}`, reward: proxyReward });
  const verifierReward = (y) => R.UTILITY.height * R.verifierReward(y);

  // ---------- colors ----------

  function readPalette(el) {
    const cs = getComputedStyle(el);
    const v = (n) => cs.getPropertyValue(n).trim();
    return {
      ink: v("--ink"), muted: v("--muted"), paper: v("--paper"), teal: v("--teal"), amber: v("--amber"),
      red: v("--red"), acceptFill: v("--accept-fill"), font: cs.fontFamily, dark: v("color-scheme") === "dark",
    };
  }
  function hexToRgb(hex) {
    const h = hex.replace("#", "");
    const n = parseInt(h.length === 3 ? [...h].map((c) => c + c).join("") : h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  // ---------- field: one canvas showing the answer space ----------

  class Field {
    constructor(canvas, layers) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.layers = layers;
      this.trails = [];
      this.lastSpawn = 0;
      this.heat = null;
      this.heatKey = "";
      const [x0, x1, y0, y1] = R.VIEW;
      canvas.style.aspectRatio = `${x1 - x0} / ${y1 - y0}`;
      new ResizeObserver(() => this.resize()).observe(canvas);
    }

    resize() {
      const dpr = window.devicePixelRatio || 1;
      const [x0, x1, y0, y1] = R.VIEW;
      this.width = this.canvas.clientWidth;
      this.height = this.canvas.clientHeight;
      if (!this.width) return;
      this.canvas.width = Math.round(this.width * dpr);
      this.canvas.height = Math.round(this.height * dpr);
      this.scale = Math.min(this.width / (x1 - x0), this.height / (y1 - y0));
      this.ox = this.width / 2 - (this.scale * (x0 + x1)) / 2;
      this.oy = this.height / 2 + (this.scale * (y0 + y1)) / 2;
      this.heatKey = "";
      this.render(performance.now());
    }

    toPx(q) {
      return [this.ox + q[0] * this.scale, this.oy - q[1] * this.scale];
    }

    toWorld(x, y) {
      return [(x - this.ox) / this.scale, (this.oy - y) / this.scale];
    }

    clearWalks() {
      this.trails = [];
    }

    spawn(now) {
      const policy = this.layers.policy && this.layers.policy();
      if (!policy) return;
      this.trails.push({ points: R.sampleWalk(policy, normal), born: now });
      if (this.trails.length > 80) this.trails.shift();
    }

    tick(now) {
      if (now - this.lastSpawn > SPAWN_MS) {
        this.spawn(now);
        this.lastSpawn = now;
      }
      this.trails = this.trails.filter((t) => now - t.born < GROW_MS + FADE_MS);
      this.render(now);
    }

    // a static set of finished walks for readers who turned off motion
    fillStatic() {
      this.trails = [];
      for (let i = 0; i < 30; i++) this.spawn(-1e9);
    }

    heatmap(pal) {
      const src = this.layers.heatmap && this.layers.heatmap();
      if (!src) return null;
      const key = `${src.key}|${this.width}|${pal.teal}`;
      if (key === this.heatKey) return this.heat;
      const cols = 110;
      const rows = Math.max(1, Math.round((cols * this.height) / this.width));
      const values = new Float32Array(cols * rows);
      for (let j = 0; j < rows; j++) {
        for (let i = 0; i < cols; i++) {
          const x = (this.width * (i + 0.5)) / cols;
          const y = (this.height * (j + 0.5)) / rows;
          values[j * cols + i] = src.reward([(x - this.ox) / this.scale, (this.oy - y) / this.scale]);
        }
      }
      // clip to percentiles so one far corner of linear extrapolation does not wash out the rest
      const sorted = Float32Array.from(values).sort();
      const lo = sorted[Math.floor(0.03 * sorted.length)];
      const hi = sorted[Math.floor(0.99 * sorted.length)];
      const off = document.createElement("canvas");
      off.width = cols;
      off.height = rows;
      const octx = off.getContext("2d");
      const img = octx.createImageData(cols, rows);
      const [r, g, b] = hexToRgb(pal.teal);
      values.forEach((v, k) => {
        const t = Math.min(1, Math.max(0, (v - lo) / (hi - lo || 1)));
        img.data.set([r, g, b, Math.round(255 * 0.6 * t ** 1.3)], 4 * k);
      });
      octx.putImageData(img, 0, 0);
      this.heat = off;
      this.heatKey = key;
      return off;
    }

    circle(center, radius, stroke, opts = {}) {
      const ctx = this.ctx;
      const [x, y] = this.toPx(center);
      ctx.beginPath();
      ctx.arc(x, y, radius * this.scale, 0, 2 * Math.PI);
      if (opts.fill) {
        ctx.fillStyle = opts.fill;
        ctx.fill();
      }
      ctx.setLineDash(opts.dash || []);
      ctx.strokeStyle = stroke;
      ctx.lineWidth = opts.width || 1;
      ctx.globalAlpha = opts.alpha ?? 1;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }

    dot(q, r, fill, stroke) {
      const ctx = this.ctx;
      const [x, y] = this.toPx(q);
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      if (fill) {
        ctx.fillStyle = fill;
        ctx.fill();
      }
      if (stroke) {
        ctx.strokeStyle = stroke;
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }
    }

    label(text, q, pal, opts = {}) {
      const ctx = this.ctx;
      const [x, y] = this.toPx(q);
      ctx.font = `${opts.italic ? "italic " : ""}12px ${pal.font}`;
      ctx.fillStyle = opts.color || pal.muted;
      ctx.textAlign = opts.align || "center";
      ctx.fillText(text, x + (opts.dx || 0), y + (opts.dy || 0));
    }

    drawUtility(pal) {
      const { center, width, height } = R.UTILITY;
      [0.75, 1.5, 2.25].forEach((level) => {
        const r = width * Math.sqrt(2 * Math.log(height / level));
        this.circle(center, r, pal.ink, { dash: [2, 4], alpha: 0.5 });
      });
      const outer = width * Math.sqrt(2 * Math.log(height / 0.75));
      this.label("hidden utility", [center[0], center[1] + outer], pal, { dy: -6 });
    }

    drawPolicy(policy, pal) {
      const mean = R.policyMean(policy);
      const sigma = Math.exp(policy.logSigma);
      const [x0, x1, y0, y1] = R.VIEW;
      const inside = mean[0] > x0 && mean[0] < x1 && mean[1] > y0 && mean[1] < y1;
      if (inside) {
        this.circle(mean, 2 * sigma, pal.teal, { width: 2 });
        return;
      }
      // the policy ran off the plot, so point at where it went
      const edge = [Math.min(x1 - 0.08, Math.max(x0 + 0.08, mean[0])), Math.min(y1 - 0.08, Math.max(y0 + 0.08, mean[1]))];
      const [ex, ey] = this.toPx(edge);
      const ang = Math.atan2(-(mean[1] - R.START[1]), mean[0] - R.START[0]);
      const ctx = this.ctx;
      ctx.save();
      ctx.translate(ex, ey);
      ctx.rotate(ang);
      ctx.fillStyle = pal.red;
      ctx.beginPath();
      ctx.moveTo(10, 0);
      ctx.lineTo(-6, -7);
      ctx.lineTo(-6, 7);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
      // below the arrow, so it clears labels that sit along the top edge
      this.label("policy left the plot", edge, pal, { color: pal.red, align: "right", dx: -4, dy: 22 });
    }

    render(now) {
      if (!this.width) return;
      const pal = readPalette(this.canvas);
      const ctx = this.ctx;
      const dpr = window.devicePixelRatio || 1;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, this.width, this.height);
      ctx.lineJoin = "round";

      const heat = this.heatmap(pal);
      if (heat) {
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(heat, 0, 0, this.width, this.height);
      }
      if (this.layers.verifier) {
        this.circle(R.VERIFIER.center, R.VERIFIER.r, pal.red, { fill: pal.acceptFill, alpha: 0.7 });
        this.label("verifier accepts", [R.VERIFIER.center[0], R.VERIFIER.center[1] - R.VERIFIER.r], pal, { color: pal.red, dy: 15 });
      }
      if (this.layers.utility && this.layers.utility()) this.drawUtility(pal);
      if (this.layers.lengthAxis) {
        const [x0, x1, y0] = R.VIEW;
        this.label("longer answers →", [x1 - 0.05, y0 + 0.06], pal, { align: "right" });
      }
      if (this.layers.ref) {
        this.circle(R.REF_MEAN, 2 * R.REF_SIGMA, pal.muted, { dash: [5, 4], alpha: 0.8 });
        this.label("base model", [R.REF_MEAN[0], R.REF_MEAN[1] - 2 * R.REF_SIGMA], pal, { dy: 15 });
      }
      if (this.layers.extra) this.layers.extra(this, pal);

      ctx.lineWidth = 1;
      for (const t of this.trails) {
        const age = now - t.born;
        const shown = reducedMotion || t.born < 0 ? W.STEPS + 1 : Math.min(W.STEPS + 1, 1 + Math.floor((W.STEPS * age) / GROW_MS));
        const fade = reducedMotion || t.born < 0 ? 0.6 : Math.max(0, 1 - Math.max(0, age - GROW_MS) / FADE_MS);
        ctx.globalAlpha = 0.4 * fade;
        ctx.strokeStyle = pal.ink;
        ctx.beginPath();
        for (let i = 0; i < shown; i++) {
          const [x, y] = this.toPx(t.points[i]);
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.stroke();
        if (shown > W.STEPS) {
          ctx.globalAlpha = 0.8 * fade;
          this.dot(t.points[W.STEPS], 2, pal.ink);
        }
      }
      ctx.globalAlpha = 1;

      const policy = this.layers.policy && this.layers.policy();
      if (policy && this.layers.showPolicyRing) this.drawPolicy(policy, pal);
      this.dot(R.START, 5, pal.ink);
      this.label("prompt x", R.START, pal, { dy: 20, italic: false });
    }
  }

  // ---------- slider helpers ----------

  const BETA_MIN = Math.log(0.03);
  const BETA_MAX = Math.log(3);
  const betaOf = (slider) => Math.exp(BETA_MIN + ((BETA_MAX - BETA_MIN) * slider.value) / 100);
  const sliderOf = (beta) => (100 * (Math.log(beta) - BETA_MIN)) / (BETA_MAX - BETA_MIN);
  const fmtBeta = (b) => (b >= 0.1 ? b.toFixed(2) : b.toPrecision(2));

  function gains(policy) {
    const now = R.evaluate(policy, store.model, 500);
    const base = R.evaluate(R.refPolicy(), store.model, 500);
    return { ...now, proxyGain: now.proxy - base.proxy, trueGain: now.utility - base.utility };
  }

  // ---------- panel: preferences and reward model ----------

  function mountPrefs(panel) {
    const showUtility = $("#prefs-utility", panel);
    let pair = null;
    const newPair = () => {
      const ref = R.refPolicy();
      const wa = R.sampleWalk(ref, normal);
      const wb = R.sampleWalk(ref, normal);
      pair = { wa, wb, a: wa[W.STEPS], b: wb[W.STEPS] };
    };
    newPair();

    const field = new Field($("#prefs-field", panel), {
      heatmap: () => (store.pairs.length >= 3 ? storeHeat() : null),
      utility: () => showUtility.checked,
      extra: (f, pal) => {
        const ctx = f.ctx;
        // past labels fade as they pile up, so the pair being judged stays the clearest thing on the canvas
        const fadeBy = Math.min(1, 40 / Math.max(1, store.pairs.length));
        for (const p of store.pairs) {
          const [ax, ay] = f.toPx(p.a);
          const [bx, by] = f.toPx(p.b);
          ctx.strokeStyle = pal.ink;
          ctx.globalAlpha = 0.03 + 0.11 * fadeBy;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();
          ctx.globalAlpha = 0.3 + 0.55 * fadeBy;
          f.dot(p.winner === 0 ? p.a : p.b, 2.2, pal.teal);
          f.dot(p.winner === 0 ? p.b : p.a, 1.9, pal.paper, pal.muted);
        }
        ctx.globalAlpha = 1;
        [[pair.wa, pal.ink, "A"], [pair.wb, pal.amber, "B"]].forEach(([w, color, name]) => {
          const trace = () => {
            ctx.beginPath();
            w.forEach((q, i) => {
              const [x, y] = f.toPx(q);
              i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
            });
            ctx.stroke();
          };
          // a paper-colored halo separates the walk from the dots under it
          ctx.strokeStyle = pal.paper;
          ctx.lineWidth = 5;
          trace();
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          trace();
          f.dot(w[W.STEPS], 6.5, pal.paper);
          f.dot(w[W.STEPS], 5, color);
          ctx.font = `600 13px ${pal.font}`;
          ctx.fillStyle = pal.paper;
          const [lx, ly] = f.toPx(w[W.STEPS]);
          ctx.textAlign = "left";
          ctx.lineWidth = 4;
          ctx.strokeStyle = pal.paper;
          ctx.strokeText(name, lx + 10, ly + 4);
          ctx.fillStyle = color;
          ctx.fillText(name, lx + 10, ly + 4);
        });
      },
    });

    const youAgree = () => {
      const mine = store.pairs.filter((p) => p.source === "you");
      if (!mine.length) return NaN;
      return mine.filter((p) => (R.trueUtility(p.a) >= R.trueUtility(p.b) ? 0 : 1) === p.winner).length / mine.length;
    };

    function refresh() {
      const ai = store.pairs.filter((q) => q.source === "ai").length;
      $("#prefs-count", panel).textContent = ai ? `${store.pairs.length} (${ai} from the AI labeler)` : store.pairs.length;
      $("#prefs-acc", panel).textContent = store.pairs.length >= 3 ? pct(R.pairAccuracy(store.model, HELD_OUT)) : "–";
      $("#prefs-agree", panel).textContent = pct(youAgree());
      field.render(performance.now());
    }

    const choose = (winner) => {
      setPairs(store.pairs.concat([{ a: pair.a, b: pair.b, winner, source: "you" }]));
      newPair();
      refresh();
    };
    panel.addEventListener("click", (e) => {
      const act = e.target.closest("[data-act]")?.dataset.act;
      if (act === "prefer-a") choose(0);
      if (act === "prefer-b") choose(1);
      if (act === "skip-pair") { newPair(); refresh(); }
      if (act === "auto-50") setPairs(store.pairs.concat(autoPairs(50)));
      if (act === "ai-50") setPairs(store.pairs.concat(aiPairs(50)));
      if (act === "reset-labels") setPairs([]);
    });
    document.addEventListener("keydown", (e) => {
      if (panel.hidden || e.target.closest("input, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "a" || e.key === "A") choose(0);
      if (e.key === "b" || e.key === "B") choose(1);
    });
    showUtility.addEventListener("change", refresh);
    const noiseInput = $("#prefs-noise", panel);
    noiseInput.addEventListener("input", () => {
      labeler.noise = +noiseInput.value;
      $("#prefs-noise-out", panel).textContent = `${labeler.noise}×`;
    });

    // the hidden utility's peak can be dragged while it is shown; every step then scores against the new peak
    const canvas = field.canvas;
    let dragging = false;
    const local = (e) => {
      const r = canvas.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    };
    const nearPeak = (x, y) => {
      if (!showUtility.checked) return false;
      const [px, py] = field.toPx(R.UTILITY.center);
      return Math.hypot(px - x, py - y) < 24;
    };
    canvas.addEventListener("pointerdown", (e) => {
      const [x, y] = local(e);
      if (!nearPeak(x, y)) return;
      dragging = true;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
      const [x, y] = local(e);
      canvas.style.cursor = dragging ? "grabbing" : nearPeak(x, y) ? "grab" : "";
      if (!dragging) return;
      const [x0, x1, y0, y1] = R.VIEW;
      const q = field.toWorld(x, y);
      // move the array in place, since the verifier region shares it
      R.UTILITY.center[0] = Math.max(x0 + 0.3, Math.min(x1 - 0.2, q[0]));
      R.UTILITY.center[1] = Math.max(y0 + 0.2, Math.min(y1 - 0.2, q[1]));
      field.render(performance.now());
    });
    const release = () => {
      if (!dragging) return;
      dragging = false;
      $("#prefs-moved", panel).hidden = !store.pairs.length;
      // refitting bumps the version, so later steps restart against the moved peak
      setPairs(store.pairs);
    };
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", release);

    store.listeners.push(refresh);
    refresh();
    return { tick: () => {} };
  }

  // ---------- panel: rl with a kl penalty, plus the overoptimization sweep ----------

  function mountRL(panel) {
    const slider = $("#rl-beta", panel);
    const showUtility = $("#rl-utility", panel);
    const trainBtn = $("[data-act=rl-train]", panel);
    slider.value = sliderOf(0.3);
    let trainer = null;
    let history = null;
    let target = 0;
    let running = false;
    let version = -1;
    let sweepVersion = -1;
    const gainSeries = [
      { name: "proxy reward (what RL sees)", color: "--amber" },
      { name: "true utility (what people want)", color: "--teal" },
    ];
    const klSeries = [{ name: "KL to base model", color: "--slate" }];

    const field = new Field($("#rl-field", panel), {
      heatmap: () => (store.pairs.length >= MIN_PAIRS ? storeHeat() : null),
      utility: () => showUtility.checked,
      ref: true,
      showPolicyRing: true,
      policy: () => (trainer ? trainer.policy : R.refPolicy()),
    });
    const gainChart = C.lineChart($("#rl-chart-gain", panel), {
      x: [], series: [
        { name: "proxy reward (what RL sees)", color: "--amber", values: [] },
        { name: "true utility (what people want)", color: "--teal", values: [] },
      ],
      xLabel: "training step", yFormat: (v) => fmt(v, 1), empty: "Press Train to start",
      tooltip: (i) => `step ${history.step[i]}<br>proxy gain <b>${fmt(history.proxy[i])}</b><br>true gain <b>${fmt(history.util[i])}</b>`,
    });
    const klChart = C.lineChart($("#rl-chart-kl", panel), {
      x: [], series: [{ name: "KL to base model", color: "--slate", values: [] }],
      xLabel: "training step", yFormat: (v) => fmt(v, 1), empty: "",
      tooltip: (i) => `step ${history.step[i]}<br>KL <b>${fmt(history.kl[i])}</b>`,
    });
    let sweep = [];
    const sweepChart = C.lineChart($("#sweep-chart", panel), {
      x: [], markers: true, series: [
        { name: "proxy reward gain", color: "--amber", values: [] },
        { name: "true utility gain", color: "--teal", values: [] },
      ],
      xLabel: "√KL to base model (lower β to the right)", xFormat: (v) => fmt(v, 1), yFormat: (v) => fmt(v, 1),
      tooltip: (i) => `β = ${fmtBeta(sweep[i].beta)}, KL ${fmt(sweep[i].kl, 1)}<br>proxy gain <b>${fmt(sweep[i].proxyGain)}</b><br>true gain <b>${fmt(sweep[i].trueGain)}</b>`,
    });

    const batchInput = $("#rl-batch", panel);
    const lrInput = $("#rl-lr", panel);
    const LR_MIN = Math.log(0.003);
    const LR_MAX = Math.log(0.5);
    const lrOf = () => Math.exp(LR_MIN + ((LR_MAX - LR_MIN) * lrInput.value) / 100);
    lrInput.value = (100 * (Math.log(0.01) - LR_MIN)) / (LR_MAX - LR_MIN);
    let fromPolicy = 0;

    function reset() {
      $("#rl-batch-out", panel).textContent = batchInput.value;
      $("#rl-lr-out", panel).textContent = lrOf().toPrecision(2);
      trainer = R.reinforceTrainer({
        reward: proxyReward, beta: betaOf(slider), seed: (Math.random() * 1e9) | 0, batch: +batchInput.value, lr: lrOf(),
      });
      history = { step: [], proxy: [], util: [], kl: [] };
      target = 0;
      running = false;
      field.clearWalks();
      record(0);
      showStats();
    }
    function record(step) {
      const g = gains(trainer.policy);
      history.step.push(step);
      history.proxy.push(g.proxyGain);
      history.util.push(g.trueGain);
      history.kl.push(g.kl);
      return g;
    }
    function showStats() {
      const g = gains(trainer.policy);
      $("#rl-beta-out", panel).textContent = fmtBeta(betaOf(slider));
      $("#rl-step", panel).textContent = history.step[history.step.length - 1];
      $("#rl-kl", panel).textContent = fmt(g.kl);
      $("#rl-sigma", panel).textContent = `${pct(g.sigma / R.REF_SIGMA)}`;
      $("#rl-true", panel).textContent = fmt(g.trueGain);
      $("#rl-proxy", panel).textContent = fmt(g.proxyGain);
      trainBtn.textContent = running ? "Pause" : history.step.length > 1 ? "Train 300 more" : "Train 300 steps";
      gainChart.update({ x: history.step, series: [{ ...gainSeries[0], values: history.proxy }, { ...gainSeries[1], values: history.util }] });
      klChart.update({ x: history.step, series: [{ ...klSeries[0], values: history.kl }] });
      if (sweep.length) sweepChart.update({ highlight: nearestBeta() });
      if (reducedMotion) field.fillStatic();
      field.render(performance.now());
    }

    function nearestBeta() {
      const lb = Math.log(betaOf(slider));
      let best = 0;
      sweep.forEach((s, i) => {
        if (Math.abs(Math.log(s.beta) - lb) < Math.abs(Math.log(sweep[best].beta) - lb)) best = i;
      });
      return best;
    }

    function runSweep() {
      if (sweepVersion === store.version || store.pairs.length < MIN_PAIRS) return;
      sweepVersion = store.version;
      $("#sweep-status", panel).textContent = "Computing the sweep…";
      setTimeout(() => {
        const base = R.evaluate(R.refPolicy(), store.model, 2000);
        sweep = R.betaSweep(store.model).map((s) => ({ ...s, proxyGain: s.proxy - base.proxy, trueGain: s.utility - base.utility }));
        const order = sweep.map((_, i) => i).sort((i, j) => sweep[i].kl - sweep[j].kl);
        sweep = order.map((i) => sweep[i]);
        sweepChart.update({
          x: sweep.map((s) => Math.sqrt(s.kl)),
          series: [{ ...gainSeries[0], name: "proxy reward gain", values: sweep.map((s) => s.proxyGain) },
                   { ...gainSeries[1], name: "true utility gain", values: sweep.map((s) => s.trueGain) }],
          highlight: nearestBeta(),
        });
        const best = sweep.reduce((a, b) => (b.trueGain > a.trueGain ? b : a));
        const last = sweep[sweep.length - 1];
        $("#sweep-status", panel).textContent = best === last
          ? `With ${store.pairs.length} labels, the true utility still rises at the smallest β. Collect more labels to see the turn.`
          : `With ${store.pairs.length} labels, the true utility peaks at β ≈ ${fmtBeta(best.beta)}. At lower β, the proxy keeps rising and the true utility falls.`;
      }, 30);
    }

    trainBtn.addEventListener("click", () => {
      if (running) running = false;
      else {
        target = history.step[history.step.length - 1] + TRAIN_STEPS;
        running = true;
      }
      showStats();
    });
    $("[data-act=rl-reset]", panel).addEventListener("click", reset);
    slider.addEventListener("input", reset);
    batchInput.addEventListener("input", reset);
    lrInput.addEventListener("input", reset);
    $("[data-act=rl-collect]", panel).addEventListener("click", () => {
      if (!trainer) return;
      setPairs(store.pairs.concat(autoPairs(100, trainer.policy)));
      fromPolicy += 100;
      // keep training the same policy against the refit reward model instead of restarting
      version = store.version;
      $("#rl-collected", panel).textContent = `${fromPolicy} labels collected from the policy so far.`;
      runSweep();
      field.render(performance.now());
      showStats();
    });
    showUtility.addEventListener("change", () => field.render(performance.now()));

    function onShow() {
      if (version !== store.version) {
        version = store.version;
        reset();
      }
      runSweep();
    }
    return {
      onShow,
      tick(now) {
        if (running) {
          let step = history.step[history.step.length - 1];
          for (let k = 0; k < 3 && step < target; k++) {
            trainer.step();
            step++;
            record(step);
          }
          if (step >= target) running = false;
          showStats();
        }
        if (!reducedMotion) field.tick(now);
      },
    };
  }

  // ---------- two-field comparison panels: dpo vs rlhf, rlvr vs rlhf ----------

  function mountCompare(panel, cfg) {
    // either one shared beta slider, or one per side marked with data-side
    const sliders = [...panel.querySelectorAll(".beta")];
    const sliderFor = (side) => sliders.find((s) => s.dataset.side === side.key) || sliders[0];
    const showUtility = $(".show-utility", panel);
    const trainBtn = $("[data-act=compare-train]", panel);
    sliders.forEach((s) => (s.value = sliderOf(cfg.defaultBeta)));
    let sides = null;
    let step = 0;
    let running = false;
    let version = -1;

    const fields = cfg.sides.map((side) =>
      new Field($(side.canvas, panel), {
        heatmap: () => (side.heatmap && store.pairs.length >= MIN_PAIRS ? storeHeat() : null),
        verifier: side.verifier,
        utility: () => showUtility.checked,
        ref: true,
        showPolicyRing: true,
        policy: () => (sides ? sides[cfg.sides.indexOf(side)].policy : R.refPolicy()),
        extra: side.pairs ? (f, pal) => {
          f.ctx.globalAlpha = 0.6;
          for (const p of store.pairs) {
            f.dot(p.winner === 0 ? p.a : p.b, 2, pal.teal);
            f.dot(p.winner === 0 ? p.b : p.a, 1.8, pal.paper, pal.muted);
          }
          f.ctx.globalAlpha = 1;
        } : null,
      }),
    );

    const spreadChart = cfg.spreadChart && C.lineChart($(cfg.spreadChart, panel), {
      x: [], yMax: 1.2, yFormat: pct, xLabel: "training step", empty: "Press Train both",
      series: [], tooltip: (i) => `step ${spreadHistory.step[i]}<br>` + cfg.sides.map((sd, j) => `${sd.label} <b>${pct(spreadHistory.values[j][i])}</b>`).join("<br>"),
    });
    let spreadHistory = { step: [], values: cfg.sides.map(() => []) };
    function recordSpread() {
      if (!spreadChart) return;
      spreadHistory.step.push(step);
      sides.forEach((s, j) => spreadHistory.values[j].push(Math.exp(s.policy.logSigma) / R.REF_SIGMA));
      spreadChart.update({
        x: spreadHistory.step,
        series: cfg.sides.map((sd, j) => ({ name: sd.label, color: j === 0 ? "--amber" : "--teal", values: spreadHistory.values[j] })),
      });
    }

    function reset() {
      sides = cfg.sides.map((side) => side.trainer(betaOf(sliderFor(side))));
      spreadHistory = { step: [], values: cfg.sides.map(() => []) };
      step = 0;
      running = false;
      fields.forEach((f) => f.clearWalks());
      show();
    }
    function show() {
      sliders.forEach((s) => {
        const out = panel.querySelector(s.dataset.side ? `.beta-out[data-side="${s.dataset.side}"]` : ".beta-out");
        if (out) out.textContent = fmtBeta(betaOf(s));
      });
      trainBtn.textContent = running ? `Training… ${step}/${TRAIN_STEPS}` : step ? "Retrain" : `Train both, ${TRAIN_STEPS} steps`;
      trainBtn.disabled = running;
      cfg.sides.forEach((side, i) => {
        const g = gains(sides[i].policy);
        for (const [key, value] of Object.entries(cfg.readout(g))) {
          const cell = $(`[data-out="${side.key}-${key}"]`, panel);
          if (cell) cell.textContent = value;
        }
      });
      if (reducedMotion) fields.forEach((f) => f.fillStatic());
      fields.forEach((f) => f.render(performance.now()));
    }

    trainBtn.addEventListener("click", () => {
      if (step) reset();
      running = true;
      show();
    });
    sliders.forEach((s) => s.addEventListener("input", reset));
    showUtility.addEventListener("change", () => fields.forEach((f) => f.render(performance.now())));
    const newLabels = $("[data-act=new-labels]", panel);
    if (newLabels) newLabels.addEventListener("click", () => {
      setPairs(autoPairs(Math.max(200, store.pairs.length)));
      version = store.version;
      reset();
      running = true;
      show();
    });

    return {
      reset,
      onShow() {
        if (version !== store.version) {
          version = store.version;
          reset();
        }
      },
      tick(now) {
        if (running) {
          for (let k = 0; k < 5 && step < TRAIN_STEPS; k++) {
            sides.forEach((s) => s.step());
            step++;
          }
          recordSpread();
          if (step >= TRAIN_STEPS) running = false;
          show();
        }
        if (!reducedMotion) fields.forEach((f) => f.tick(now));
      },
    };
  }

  // ---------- panel: reward hacking and defenses ----------

  function mountHacking(panel) {
    const HACK_SEED = 7;
    const SPUR_BETA = 0.3;
    const ROUNDS = 10;
    const ROUND_STEPS = 15;
    const showUtility = $(".show-utility", panel);
    const biasSlider = $("#hack-bias", panel);
    const defBeta = $("#def-beta", panel);
    defBeta.value = sliderOf(0.1);
    const utilityOn = () => showUtility.checked;
    const refUtility = R.evaluate(R.refPolicy(), null, 1000).utility;
    const pe = (policy) => {
      const e = R.evaluate(policy, null, 1000);
      return { ...e, gain: e.utility - refUtility, dx: R.policyMean(policy)[0] - R.UTILITY.center[0] };
    };
    const setOut = (key, g, extra = {}) => {
      const vals = { true: fmt(g.gain), dx: fmt(g.dx), kl: fmt(g.kl), ...extra };
      for (const [k, v] of Object.entries(vals)) {
        const cell = $(`[data-out="${key}-${k}"]`, panel);
        if (cell) cell.textContent = v;
      }
    };
    const pairDots = (getPairs) => (f, pal) => {
      const pairs = getPairs();
      if (!pairs) return;
      f.ctx.globalAlpha = 0.55;
      for (const p of pairs) {
        f.dot(p.winner === 0 ? p.a : p.b, 1.8, pal.teal);
        f.dot(p.winner === 0 ? p.b : p.a, 1.6, pal.paper, pal.muted);
      }
      f.ctx.globalAlpha = 1;
    };

    // part a: the same rlhf with fair and with length-loving labelers
    const spur = ["fair", "biased"].map((kind) => {
      const bias = kind === "fair" ? 0 : R.LENGTH_BIAS;
      const pairs = R.collectPairs(300, HACK_SEED, { bias });
      const model = R.fitRewardModel(pairs);
      const side = { kind, bias, pairs, model, trainer: null, run: 0 };
      side.reset = () => {
        side.trainer = R.reinforceTrainer({ reward: (y) => R.rmReward(model, y), beta: SPUR_BETA, seed: 5 });
        side.step = 0;
      };
      side.reset();
      side.field = new Field($(`#hack-field-${kind}`, panel), {
        heatmap: () => ({ key: `spur-${kind}`, reward: (y) => R.rmReward(model, y) }),
        utility: utilityOn,
        lengthAxis: true,
        ref: true,
        showPolicyRing: true,
        policy: () => side.trainer.policy,
      });
      return side;
    });
    let spurRunning = false;
    const spurBtn = $("[data-act=spur-train]", panel);
    function showSpur() {
      spur.forEach((side) => {
        setOut(`spur-${side.kind}`, pe(side.trainer.policy), { acc: pct(R.pairAccuracy(side.model, HELD_OUT)) });
      });
      spurBtn.textContent = spurRunning ? `Training… ${spur[0].step}/${TRAIN_STEPS}` : spur[0].step ? "Retrain" : `Train both, ${TRAIN_STEPS} steps`;
      spurBtn.disabled = spurRunning;
    }
    spurBtn.addEventListener("click", () => {
      spur.forEach((s) => { s.reset(); s.field.clearWalks(); });
      spurRunning = true;
      showSpur();
    });

    // part b: plain rlhf, an ensemble, and iterated rlhf, all at a low beta
    const def = { plain: null, ensemble: null, iterated: null, round: 0, step: 0, running: false, iterModel: null, iterPairs: null };
    function resetDefenses() {
      const bias = +biasSlider.value;
      const pairs = R.collectPairs(300, HACK_SEED, { bias });
      const model = R.fitRewardModel(pairs);
      def.beta = betaOf(defBeta);
      const members = R.fitEnsemble(pairs, +$("#def-members", panel).value, HACK_SEED);
      const cautious = R.conservativeReward(members, +$("#def-caution", panel).value);
      $("#def-beta-out", panel).textContent = fmtBeta(def.beta);
      $("#def-members-out", panel).textContent = $("#def-members", panel).value;
      $("#def-caution-out", panel).textContent = $("#def-caution", panel).value;
      def.bias = bias;
      def.model = model;
      def.cautious = cautious;
      def.plain = R.reinforceTrainer({ reward: (y) => R.rmReward(model, y), beta: def.beta, seed: 5 });
      def.ensemble = R.reinforceTrainer({ reward: cautious, beta: def.beta, seed: 5 });
      def.iterPolicy = R.refPolicy();
      def.iterModel = model;
      def.iterPairs = pairs;
      def.round = 0;
      def.step = 0;
      def.running = false;
      def.version = (def.version || 0) + 1;
      defFields.forEach((f) => f.clearWalks());
      showDef();
    }
    const defFields = [
      new Field($("#hack-field-plain", panel), {
        heatmap: () => def.model && { key: `plain-${def.version}`, reward: (y) => R.rmReward(def.model, y) },
        utility: utilityOn, lengthAxis: true, ref: true, showPolicyRing: true,
        policy: () => def.plain && def.plain.policy,
      }),
      new Field($("#hack-field-ensemble", panel), {
        heatmap: () => def.cautious && { key: `ens-${def.version}`, reward: def.cautious },
        utility: utilityOn, lengthAxis: true, ref: true, showPolicyRing: true,
        policy: () => def.ensemble && def.ensemble.policy,
      }),
      new Field($("#hack-field-iterated", panel), {
        heatmap: () => def.iterModel && { key: `iter-${def.version}-${def.round}`, reward: (y) => R.rmReward(def.iterModel, y) },
        utility: utilityOn, lengthAxis: true, ref: true, showPolicyRing: true,
        policy: () => def.iterPolicy,
        extra: pairDots(() => (def.round ? def.iterPairs : null)),
      }),
    ];
    const defBtn = $("[data-act=def-train]", panel);
    function showDef() {
      $("#hack-bias-out", panel).textContent = (+biasSlider.value).toFixed(1);
      if (!def.plain) return;
      setOut("def-plain", pe(def.plain.policy));
      setOut("def-ensemble", pe(def.ensemble.policy));
      setOut("def-iterated", pe(def.iterPolicy));
      $("#hack-round", panel).textContent = def.round ? `round ${def.round} of ${ROUNDS}` : "not started";
      defBtn.textContent = def.running ? "Running…" : def.step ? "Run again" : "Run all three";
      defBtn.disabled = def.running;
      if (reducedMotion) defFields.forEach((f) => f.fillStatic());
      defFields.forEach((f) => f.render(performance.now()));
    }
    defBtn.addEventListener("click", () => {
      if (def.step) resetDefenses();
      def.running = true;
      showDef();
    });
    biasSlider.addEventListener("input", resetDefenses);
    ["#def-beta", "#def-members", "#def-caution"].forEach((id) => $(id, panel).addEventListener("input", resetDefenses));
    showUtility.addEventListener("change", () => [...spur.map((s) => s.field), ...defFields].forEach((f) => f.render(performance.now())));
    // 2 is where the shared bias already defeats the ensemble but iterated rlhf still works
    biasSlider.value = 2;

    let started = false;
    return {
      onShow() {
        if (started) return;
        started = true;
        resetDefenses();
        showSpur();
      },
      tick(now) {
        if (spurRunning) {
          for (let k = 0; k < 5 && spur[0].step < TRAIN_STEPS; k++) spur.forEach((s) => { s.trainer.step(); s.step++; });
          if (spur[0].step >= TRAIN_STEPS) spurRunning = false;
          showSpur();
        }
        if (def.running) {
          for (let k = 0; k < 5 && def.step < TRAIN_STEPS; k++) {
            def.plain.step();
            def.ensemble.step();
            def.step++;
          }
          // one round of fresh labels every 30 steps, so all three finish together
          if (def.step >= (def.round + 1) * (TRAIN_STEPS / ROUNDS) && def.round < ROUNDS) {
            const out = R.iterateRound(def.iterPolicy, { beta: def.beta, bias: def.bias, steps: ROUND_STEPS, seed: HACK_SEED * 100 + def.round + 1 });
            def.iterPolicy = out.policy;
            def.iterModel = out.model;
            def.iterPairs = out.pairs;
            def.round++;
          }
          if (def.step >= TRAIN_STEPS && def.round >= ROUNDS) def.running = false;
          showDef();
        }
        if (!reducedMotion) [...spur.map((s) => s.field), ...defFields].forEach((f) => f.tick(now));
      },
    };
  }

  const readout = (g) => ({ true: fmt(g.trueGain), proxy: fmt(g.proxyGain), kl: fmt(g.kl), sigma: pct(g.sigma / R.REF_SIGMA), pass: pct(g.pass) });

  // ---------- empty state and wiring ----------

  function updateEmptyStates() {
    document.querySelectorAll("[data-needs-labels]").forEach((el) => {
      el.hidden = store.pairs.length >= MIN_PAIRS;
    });
  }
  store.listeners.push(updateEmptyStates);
  document.addEventListener("click", (e) => {
    if (e.target.closest("[data-act=auto-200]")) setPairs(store.pairs.concat(autoPairs(200)));
  });

  const panels = {};
  const byId = (id) => document.getElementById(id);
  panels.prefs = mountPrefs(byId("prefs"));
  panels.rl = mountRL(byId("rl"));
  panels.dpo = mountCompare(byId("dpo"), {
    defaultBeta: 0.3,
    readout,
    sides: [
      { key: "rlhf", canvas: "#dpo-field-rlhf", heatmap: true, trainer: (beta) => R.reinforceTrainer({ reward: proxyReward, beta, seed: 5 }) },
      { key: "dpo", canvas: "#dpo-field-dpo", pairs: true, trainer: (beta) => R.dpoTrainer({ pairs: store.pairs, beta }) },
      { key: "ipo", canvas: "#dpo-field-ipo", pairs: true, trainer: (tau) => R.ipoTrainer({ pairs: store.pairs, tau }) },
    ],
  });
  panels.rlvr = mountCompare(byId("rlvr"), {
    defaultBeta: 0.3,
    readout,
    spreadChart: "#rlvr-spread",
    sides: [
      { key: "rlvr", label: "RLVR", canvas: "#rlvr-field-v", verifier: true, trainer: (beta) => R.reinforceTrainer({ reward: verifierReward, beta, seed: 5 }) },
      { key: "rlhf", label: "RLHF", canvas: "#rlvr-field-h", heatmap: true, trainer: (beta) => R.reinforceTrainer({ reward: proxyReward, beta, seed: 5 }) },
    ],
  });
  panels["hacking-rlhf"] = mountHacking(byId("hacking-rlhf"));
  const radius = $("#rlvr-radius");
  radius.addEventListener("input", () => {
    R.VERIFIER.r = +radius.value;
    $("#rlvr-radius-out").textContent = (+radius.value).toFixed(2);
    panels.rlvr.reset();
  });
  updateEmptyStates();

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
