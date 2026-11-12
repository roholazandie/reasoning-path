// canvas renderer and controls for every <figure data-variant> on the page; the math lives in walks-model.js.
(function () {
  "use strict";
  const M = window.ReasoningWalks;
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const GROW_MS = 900;
  // until the reader picks a stage, the trained figures alternate before and after on this period
  const HOLD_MS = 5000;
  const MAX_TRAILS = 120;
  const normal = M.gaussianFrom(Math.random);
  const pct = (x) => `${Math.round(100 * x)}%`;
  // trained figures spawn faster and fade sooner, so the screen shows only the current stage
  const paceOf = (variant) => (variant.train ? { every: 110, fade: 2400 } : { every: 260, fade: 5000 });

  function mount(fig) {
    const name = fig.dataset.variant;
    const base = M.VARIANTS[name];
    // copy prompts and regions so dragging or a slider never mutates the shared config
    const copyRegion = (g) => ({ ...g, center: g.center.slice(), semiAxes: g.semiAxes && g.semiAxes.slice() });
    const variant = {
      ...base,
      prompts: base.prompts.map((p) => ({ ...p, start: p.start.slice(), regions: p.regions && p.regions.map(copyRegion) })),
    };
    const canvas = fig.querySelector("canvas");
    const ctx = canvas.getContext("2d");
    const imperfectBox = fig.querySelector("input[name=imperfect]");
    const stageButtons = [...fig.querySelectorAll("[data-stage]")];
    const trainAll = () => (variant.train
      ? { imperfect: M.train(variant, true), exact: variant.imperfectToggle ? M.train(variant, false) : null }
      : null);
    let runs = trainAll();
    const roundInput = fig.querySelector("input[name=round]");
    const pace = paceOf(variant);
    // with a k slider, the figure plays best-of-k: rounds of k walks, won if any one of them lands
    const kInput = fig.dataset.kInput ? document.getElementById(fig.dataset.kInput) : null;
    const tallyEl = fig.querySelector("[data-tally]");
    const ROUND_MS = GROW_MS + 1100;
    let rounds = [];
    let lastRound = 0;
    let tally = { won: 0, played: 0 };
    const [vx0, vx1, vy0, vy1] = variant.view;
    canvas.style.aspectRatio = `${vx1 - vx0} / ${vy1 - vy0}`;

    let colors = {};
    let trails = [];
    let lastSpawn = 0;
    let spawned = 0;
    let width = 0;
    let height = 0;
    let scale = 1;
    let originX = 0;
    let originY = 0;
    let visible = false;
    let raf = 0;
    let round = variant.train && reducedMotion ? M.ROUNDS : 0;
    let stageStart = 0;
    // a round slider replaces the automatic before/after switching
    let stagePicked = reducedMotion || !!roundInput;

    const imperfect = () => !imperfectBox || imperfectBox.checked;
    const weights = () => (runs ? (imperfect() ? runs.imperfect : runs.exact)[round] : M.zeroWeights(variant));
    const toPx = (q) => [originX + q[0] * scale, originY - q[1] * scale];
    const toWorld = (x, y) => [(x - originX) / scale, (originY - y) / scale];

    function readColors() {
      const css = getComputedStyle(fig);
      const v = (n) => css.getPropertyValue(n).trim();
      colors = {
        ink: v("--ink"), red: v("--red"), paper: v("--paper"),
        regionFill: v("--region-fill"), regionLine: v("--region-line"), acceptFill: v("--accept-fill"),
        font: css.fontFamily,
      };
    }

    function spawn(now) {
      const prompt = variant.prompts[spawned++ % variant.prompts.length];
      const points = M.sampleWalk(variant, prompt, weights(), normal);
      const hit = M.verifierAccepts(M.regionsOf(variant, prompt), points[points.length - 1], imperfect());
      trails.push({ points, hit, born: now });
      if (trails.length > MAX_TRAILS) trails.shift();
    }

    function reset() {
      trails = [];
      rounds = [];
      lastRound = 0;
      tally = { won: 0, played: 0 };
      if (reducedMotion) {
        if (kInput) spawnRound(0);
        else for (let i = 0; i < 40; i++) spawn(0);
      }
      updateTally();
      render(performance.now());
    }

    function spawnRound(now) {
      const k = +kInput.value;
      const prompt = variant.prompts[0];
      const round = { born: now, won: false, counted: false, k };
      for (let i = 0; i < k; i++) {
        const points = M.sampleWalk(variant, prompt, weights(), normal);
        const hit = M.verifierAccepts(M.regionsOf(variant, prompt), points[points.length - 1], imperfect());
        const winner = hit && !round.won;
        if (hit) round.won = true;
        trails.push({ points, hit, winner, born: now, round });
      }
      rounds.push(round);
    }

    function updateTally() {
      if (!tallyEl || !kInput) return;
      const k = +kInput.value;
      const p = M.successRate(variant, variant.prompts[0], weights(), imperfect());
      const expected = 1 - (1 - p) ** k;
      const rate = tally.played ? `${tally.won} / ${tally.played} = ${pct(tally.won / tally.played)}` : "–";
      tallyEl.innerHTML =
        `<div><span>Rounds won so far, k = ${k}</span><b>${rate}</b></div>` +
        `<div><span>Expected: 1 − (1 − ${pct(p)})<sup>${k}</sup></span><b>${pct(expected)}</b></div>`;
    }

    function setStage(after) {
      round = after ? M.ROUNDS : 0;
      stageStart = performance.now();
      stageButtons.forEach((b) => b.setAttribute("aria-pressed", String((b.dataset.stage === "after") === after)));
      reset();
    }

    function dot(q, r, hollow) {
      const [x, y] = toPx(q);
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      if (hollow) {
        ctx.fillStyle = colors.paper;
        ctx.fill();
        ctx.stroke();
      } else ctx.fill();
    }

    // draws text centred at x; a part is a string, [text, "it"] for italic, or [text, "sub"] for a subscript
    function richText(parts, x, y) {
      const plain = ctx.font;
      const runsOf = parts.map((p) =>
        typeof p === "string" ? [p, plain, 0] : [p[0], p[1] === "sub" ? `italic 10px ${colors.font}` : `italic ${plain}`, p[1] === "sub" ? 3 : 0],
      );
      const total = runsOf.reduce((w, [t, f]) => ((ctx.font = f), w + ctx.measureText(t).width), 0);
      const align = ctx.textAlign;
      ctx.textAlign = "left";
      let cx = x - total / 2;
      for (const [t, f, dy] of runsOf) {
        ctx.font = f;
        ctx.fillText(t, cx, y + dy);
        cx += ctx.measureText(t).width;
      }
      ctx.font = plain;
      ctx.textAlign = align;
    }

    function tracePath(region) {
      ctx.beginPath();
      for (let i = 0; i <= 120; i++) {
        const a = (2 * Math.PI * i) / 120;
        const r = M.radiusAt(region, a);
        const [x, y] = toPx([region.center[0] + r * Math.cos(a), region.center[1] + r * Math.sin(a)]);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
    }

    function extent(region, axis, pick) {
      let best = pick === Math.max ? -Infinity : Infinity;
      for (let i = 0; i < 120; i++) {
        const a = (2 * Math.PI * i) / 120;
        const trig = axis === 0 ? Math.cos(a) : Math.sin(a);
        best = pick(best, region.center[axis] + M.radiusAt(region, a) * trig);
      }
      return best;
    }

    function drawAcceptedRegions() {
      if (!imperfect()) return;
      for (const prompt of variant.prompts) {
        const regions = M.regionsOf(variant, prompt);
        const accepted = regions.find((g) => g.kind === "accepted");
        if (!accepted) continue;
        ctx.globalAlpha = 1;
        ctx.fillStyle = colors.acceptFill;
        ctx.strokeStyle = colors.red;
        ctx.lineWidth = 1;
        tracePath(accepted);
        ctx.fill();
        ctx.globalAlpha = 0.5;
        ctx.stroke();
        const right = Math.max(...regions.map((g) => extent(g, 0, Math.max)));
        const text = "what the verifier accepts";
        const textWidth = ctx.measureText(text).width;
        let [lx, ly] = toPx([right, accepted.center[1] + 0.12]);
        lx += 8;
        // on a narrow canvas the label does not fit to the right, so it goes above the ellipse
        if (lx + textWidth > width - 4) {
          lx = Math.max(4, width - textWidth - 4);
          ly = toPx([0, extent(accepted, 1, Math.max)])[1] - 8;
        }
        ctx.globalAlpha = 0.85;
        ctx.fillStyle = colors.red;
        ctx.textAlign = "left";
        ctx.fillText(text, lx, ly);
        ctx.textAlign = "center";
      }
    }

    function drawCorrectRegions(w) {
      for (const prompt of variant.prompts) {
        const regions = M.regionsOf(variant, prompt);
        for (const g of regions) {
          if (g.kind !== "correct") continue;
          const [cx, cy] = toPx(g.center);
          ctx.globalAlpha = 1;
          ctx.setLineDash([4, 4]);
          ctx.lineWidth = 1;
          ctx.fillStyle = colors.regionFill;
          ctx.strokeStyle = colors.regionLine;
          ctx.beginPath();
          ctx.arc(cx, cy, g.r * scale, 0, 2 * Math.PI);
          if (!(variant.imperfectToggle && imperfect())) ctx.fill();
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = colors.ink;
          ctx.globalAlpha = 0.75;
          if (name === "amortize") {
            ctx.textAlign = "left";
            ctx.fillText(pct(M.successRate(variant, prompt, w, true)), cx + g.r * scale + 6, cy + 4);
            ctx.textAlign = "center";
          } else {
            // sit the label under whichever region reaches lower, so it clears the verifier outline
            const lowest = Math.min(...regions.filter((h) => h.kind === "correct" || imperfect()).map((h) => extent(h, 1, Math.min)));
            richText([["\u{1D4AE}", "it"], ["x", "sub"], " = {correct answers ", ["y", "it"], "(", ["z", "it"], ")}"], cx, toPx([0, lowest])[1] + 15);
          }
        }
      }
    }

    function drawTrails(now) {
      ctx.lineWidth = 1.1;
      for (const t of trails) {
        const age = now - t.born;
        // walks can be longer or shorter than M.STEPS when the chain-length slider moves
        const last = t.points.length - 1;
        const shown = reducedMotion ? last + 1 : Math.min(last + 1, 1 + Math.floor((last * age) / GROW_MS));
        const done = shown > last;
        if (t.round) {
          drawRoundWalk(t, age, shown, done);
          continue;
        }
        // a walk turns red only once it has finished inside the accepted region
        ctx.strokeStyle = ctx.fillStyle = t.hit && done ? colors.red : colors.ink;
        const fade = reducedMotion ? 1 : Math.max(0, 1 - Math.max(0, age - GROW_MS) / pace.fade);
        ctx.globalAlpha = (t.hit ? 0.6 : 0.4) * fade;
        ctx.beginPath();
        for (let i = 0; i < shown; i++) {
          const [x, y] = toPx(t.points[i]);
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.stroke();
        if (done) dot(t.points[last], 2.2, !t.hit);
      }
    }

    function drawRoundWalk(t, age, shown, done) {
      const fade = reducedMotion ? 1 : Math.max(0, 1 - Math.max(0, age - (ROUND_MS - 350)) / 350);
      const lit = t.hit && done;
      ctx.strokeStyle = ctx.fillStyle = lit ? colors.red : colors.ink;
      ctx.lineWidth = t.winner && done ? 2.2 : 1.1;
      ctx.globalAlpha = (t.winner && done ? 0.95 : lit ? 0.5 : 0.35) * fade;
      ctx.beginPath();
      for (let i = 0; i < shown; i++) {
        const [x, y] = toPx(t.points[i]);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      ctx.stroke();
      if (done) dot(t.points[t.points.length - 1], t.winner ? 3.2 : 2.2, !t.hit);
      ctx.lineWidth = 1.1;
    }

    // the round's verdict sits above the target once every walk in it has finished
    function drawRoundResult(now) {
      const r = rounds[rounds.length - 1];
      if (!r || (!reducedMotion && now - r.born < GROW_MS)) return;
      const g = M.regionsOf(variant, variant.prompts[0])[0];
      const [cx, cy] = toPx(g.center);
      ctx.globalAlpha = 1;
      ctx.font = `600 13px ${colors.font}`;
      ctx.fillStyle = r.won ? colors.red : colors.ink;
      ctx.textAlign = "center";
      ctx.fillText(r.won ? "round won" : `all ${r.k} missed`, cx, cy - g.r * scale - 10);
      ctx.font = `13px ${colors.font}`;
    }

    function drawPrompts() {
      ctx.globalAlpha = 1;
      ctx.fillStyle = colors.ink;
      ctx.strokeStyle = colors.ink;
      ctx.lineWidth = 1.5;
      for (const p of variant.prompts) dot(p.start, 5.5, p.isNew);
      ctx.globalAlpha = 0.75;
      ctx.fillStyle = colors.ink;
      const first = variant.prompts[0];
      const [fx, fy] = toPx(first.start);
      if (name === "amortize") {
        const tag = (text, x, y) => {
          const half = ctx.measureText(text).width / 2 + 4;
          ctx.fillText(text, Math.max(half, Math.min(width - half, x)), y);
        };
        tag("training prompts", fx, toPx(variant.prompts[3].start)[1] - 12);
        const [nx, ny] = toPx(variant.prompts[4].start);
        tag("new prompt", nx, ny - 12);
      } else {
        richText(["prompt ", ["x", "it"]], fx, fy + 20);
      }
      if (name === "walks") {
        // place the label under the arc's peak, three noise deviations down, so few walks cross it
        const k = Math.round((M.STEPS * M.HEADING) / M.TURN);
        const peak = M.DRIFT.slice(0, k).reduce((q, d) => [q[0] + d[0], q[1] + d[1]], first.start);
        const [zx, zy] = toPx([peak[0], peak[1] - 3 * Math.sqrt(2 * variant.diffusion * M.DT * k)]);
        richText(["samples of ", ["z", "it"]], zx, zy + 10);
      }
      ctx.globalAlpha = 1;
    }

    function render(now) {
      const dpr = window.devicePixelRatio || 1;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      ctx.font = `13px ${colors.font}`;
      ctx.textAlign = "center";
      ctx.lineJoin = "round";
      const w = weights();
      drawAcceptedRegions();
      drawCorrectRegions(w);
      drawTrails(now);
      if (kInput) drawRoundResult(now);
      drawPrompts();
      updateReadout(w);
    }

    const readout = fig.querySelector("[data-readout]");
    let readoutKey = "";
    // the exact landing rates for the stage on screen, so the reader does not have to count walks
    function updateReadout(w) {
      if (!readout) return;
      const acc = variant.prompts[0].regions && variant.prompts[0].regions.find((g) => g.accepted || g.kind === "accepted");
      const key = `${round}|${imperfect()}|${acc ? acc.semiAxes + acc.angle : ""}|${variant.batch}`;
      if (key === readoutKey) return;
      readoutKey = key;
      const prompt = variant.prompts[0];
      const correct = M.successRate(variant, prompt, w, imperfect(), "correct");
      const wrong = M.successRate(variant, prompt, w, imperfect(), "wrong");
      const stage = round === 0 ? "Before training" : round === M.ROUNDS ? "After training" : `After round ${round}`;
      readout.innerHTML = `<div><span>${stage}: walks on correct answers</span><b>${pct(correct)}</b></div>` +
        (imperfect() ? `<div><span>${stage}: walks on wrong answers the verifier accepts</span><b>${pct(wrong)}</b></div>` : "");
    }

    function tick(now) {
      raf = 0;
      if (!visible || reducedMotion) return;
      if (variant.train && !stagePicked) {
        if (!stageStart) stageStart = now;
        if (now - stageStart > HOLD_MS) setStage(round !== M.ROUNDS);
      }
      if (kInput) {
        if (now - lastRound > ROUND_MS) {
          spawnRound(now);
          lastRound = now;
        }
        for (const r of rounds) {
          if (!r.counted && now - r.born >= GROW_MS) {
            r.counted = true;
            tally.played++;
            if (r.won) tally.won++;
            updateTally();
          }
        }
        trails = trails.filter((t) => now - t.born < ROUND_MS);
        rounds = rounds.slice(-2);
      } else {
        if (now - lastSpawn > pace.every) {
          spawn(now);
          lastSpawn = now;
        }
        trails = trails.filter((t) => now - t.born < GROW_MS + pace.fade);
      }
      render(now);
      raf = requestAnimationFrame(tick);
    }

    function play() {
      if (!raf && visible && !reducedMotion) raf = requestAnimationFrame(tick);
      else render(performance.now());
    }

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      scale = Math.min(width / (vx1 - vx0), height / (vy1 - vy0));
      originX = width / 2 - (scale * (vx0 + vx1)) / 2;
      originY = height / 2 + (scale * (vy0 + vy1)) / 2;
      render(performance.now());
    }

    if (variant.draggable) attachDrag();

    function attachDrag() {
      const target = variant.prompts.find((p) => p.isNew);
      let dragging = false;
      let grab = [0, 0];
      const local = (e) => {
        const r = canvas.getBoundingClientRect();
        return [e.clientX - r.left, e.clientY - r.top];
      };
      const near = (x, y) => {
        const [px, py] = toPx(target.start);
        return Math.hypot(px - x, py - y) < 22;
      };
      // stop a touch on the prompt from scrolling the page
      canvas.addEventListener("touchstart", (e) => {
        const r = canvas.getBoundingClientRect();
        const t = e.touches[0];
        if (near(t.clientX - r.left, t.clientY - r.top)) e.preventDefault();
      }, { passive: false });
      canvas.addEventListener("pointerdown", (e) => {
        const [x, y] = local(e);
        if (!near(x, y)) return;
        dragging = true;
        const [px, py] = toPx(target.start);
        grab = [px - x, py - y];
        canvas.setPointerCapture(e.pointerId);
      });
      canvas.addEventListener("pointermove", (e) => {
        const [x, y] = local(e);
        canvas.style.cursor = dragging ? "grabbing" : near(x, y) ? "grab" : "";
        if (!dragging) return;
        const q = toWorld(x + grab[0], y + grab[1]);
        // keep the prompt far enough left that its target region stays on screen
        target.start = [Math.max(vx0 + 0.05, Math.min(vx1 - 1.85, q[0])), Math.max(vy0 + 0.1, Math.min(vy1 - 0.1, q[1]))];
        reset();
        play();
      });
      const release = () => { dragging = false; };
      canvas.addEventListener("pointerup", release);
      canvas.addEventListener("pointercancel", release);
    }

    stageButtons.forEach((b) => b.addEventListener("click", () => {
      stagePicked = true;
      setStage(b.dataset.stage === "after");
      play();
    }));
    if (imperfectBox) imperfectBox.addEventListener("change", () => { reset(); play(); });
    if (kInput) kInput.addEventListener("input", () => { reset(); play(); });
    if (roundInput) {
      round = +roundInput.value;
      roundInput.addEventListener("input", () => {
        round = +roundInput.value;
        const out = fig.querySelector("output[data-for=round]");
        if (out) out.textContent = round;
        reset();
        play();
      });
    }
    // sliders that change the model itself; the ones that change what training sees retrain it
    const RETRAIN = new Set(["bumpWidth", "batch", "ellLength", "ellAngle", "keepScale"]);
    fig.querySelectorAll("input[data-param]").forEach((input) => {
      const apply = () => {
        const key = input.dataset.param;
        const value = +input.value;
        const accepted = variant.prompts[0].regions && variant.prompts[0].regions.find((g) => g.kind === "accepted");
        if (key === "ellLength") accepted.semiAxes[0] = value;
        else if (key === "ellAngle") accepted.angle = (value * Math.PI) / 180;
        else variant[key] = value;
        const out = fig.querySelector(`output[data-for=${key}]`);
        if (out) out.textContent = input.dataset.format === "x" ? `${value}×` : input.value;
        if (RETRAIN.has(key)) runs = trainAll();
      };
      input.addEventListener("input", () => {
        apply();
        reset();
        play();
        fig.dispatchEvent(new CustomEvent("walks:change", { detail: { variant } }));
      });
    });
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readColors(); render(performance.now()); });
    new MutationObserver(() => { readColors(); render(performance.now()); })
      .observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; play(); }).observe(canvas);
    new ResizeObserver(resize).observe(canvas);

    readColors();
    reset();
    resize();
    if (stageButtons.length) setStage(round === M.ROUNDS);
  }

  document.querySelectorAll("figure[data-variant]").forEach(mount);
})();
