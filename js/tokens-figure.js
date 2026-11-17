// step 2: real sampled token paths from a local model, drawn as a trie that ends in answer nodes.
// the data comes from tools/build-token-tree.py; this file only lays it out and reads it.
(function () {
  "use strict";
  const panel = document.getElementById("tokens");
  if (!panel) return;
  const fig = panel.querySelector("figure.field");
  const canvas = fig.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const $ = (sel) => panel.querySelector(sel);
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const GROW_MS = 1100;
  const ASPECT = 1.6;
  const VOTE_TRIALS = 4000;
  // a value that is not exactly 0 or 1 never prints as 0% or 100%
  const pct = (x) => (x > 0 && x < 0.005 ? "<1%" : x < 1 && x >= 0.995 ? ">99%" : `${Math.round(100 * x)}%`);

  let data = null;
  let prompt = null;
  let layout = null;
  let selected = 0;
  let hovered = -1;
  let showWhatIf = false;
  let width = 0;
  let height = 0;
  let growStart = 0;
  let raf = 0;
  let colors = {};

  function readColors() {
    const css = getComputedStyle(fig);
    const v = (n) => css.getPropertyValue(n).trim();
    colors = {
      ink: v("--ink"), muted: v("--muted"), red: v("--red"), slate: v("--slate"), paper: v("--paper"),
      line: v("--line-strong"), regionLine: v("--region-line"), acceptFill: v("--accept-fill"), teal: v("--teal"),
    };
  }

  /** Return a number written as a short decimal or as a power of ten. */
  function prob(p) {
    if (p >= 0.01) return p.toFixed(2);
    const e = Math.floor(Math.log10(p));
    return `${(p / 10 ** e).toFixed(1)} × 10<sup>${e}</sup>`;
  }

  const pathProb = (tokens) => tokens.reduce((s, t) => s * t.p, 1);
  /** Return a token as readable text, naming a token that is only whitespace. */
  const readable = (t) => (/^<\|.*\|>$/.test(t) ? "the end of the answer" : t.trim() ? t.trim() : t.includes("\n") ? "a line break" : "a space");
  const answerLabel = (path) => (path.answer === null ? "none" : String(+path.answer.toFixed(4)));

  // ---------- layout ----------

  /** Build the radix tree of the sampled paths and place its segments, leaves and answer nodes. */
  function buildLayout() {
    const samples = prompt.paths.map((p, i) => ({ ...p, index: i })).filter((p) => p.kind === "sample");
    const root = { depth: 0, children: new Map(), leaves: [] };
    for (const s of samples) {
      let node = root;
      s.tokens.forEach((t, d) => {
        if (!node.children.has(t.t)) node.children.set(t.t, { depth: d + 1, children: new Map(), leaves: [] });
        node = node.children.get(t.t);
        node.leaves.push(s.index);
      });
      root.leaves.push(s.index);
    }
    // leaves in depth-first order, larger branches first, give each path its own lane
    const order = [];
    const walk = (node) => {
      const kids = [...node.children.values()].sort((a, b) => b.leaves.length - a.leaves.length);
      const ending = node.leaves.filter((i) => !kids.some((k) => k.leaves.includes(i)));
      order.push(...ending);
      kids.forEach(walk);
    };
    walk(root);
    const lane = new Map(order.map((i, k) => [i, k]));
    const laneOf = (leaves) => leaves.reduce((s, i) => s + lane.get(i), 0) / leaves.length;

    // collapse chains with one child into segments that run between branch points
    const segments = [];
    const collapse = (node, fromDepth, fromLane) => {
      let end = node;
      while (end.children.size === 1 && end.leaves.length === [...end.children.values()][0].leaves.length) {
        end = [...end.children.values()][0];
      }
      const toLane = laneOf(end.leaves);
      const hits = end.leaves.filter((i) => prompt.paths[i].correct).length;
      segments.push({ from: fromDepth, to: end.depth, fromLane, toLane, count: end.leaves.length, share: hits / end.leaves.length, leaves: end.leaves });
      end.children.forEach((child) => collapse(child, end.depth, toLane));
    };
    root.children.forEach((child) => collapse(child, 0, laneOf(root.leaves)));

    const answers = new Map();
    const addAnswer = (path, i) => {
      const key = answerLabel(path);
      if (!answers.has(key)) answers.set(key, { key, correct: path.correct, lanes: [], members: [] });
      answers.get(key).lanes.push(lane.has(i) ? lane.get(i) : lane.get(path.parent));
      answers.get(key).members.push(i);
    };
    prompt.paths.forEach((p, i) => addAnswer(p, i));
    const answerList = [...answers.values()].map((a) => ({ ...a, lane: a.lanes.reduce((s, x) => s + x, 0) / a.lanes.length }))
      .sort((a, b) => a.lane - b.lane);
    // the axis fits the drawn samples; a longer what-if tail is clamped at the right edge
    const maxDepth = Math.max(...samples.map((s) => s.tokens.length));
    return { samples, lane, laneCount: order.length, rootLane: laneOf(root.leaves), segments, answers: answerList, maxDepth };
  }

  // pixel geometry depends on the canvas size, so it is recomputed on every resize
  function geometry() {
    const left = 30;
    const right = width - 92;
    const top = 26;
    const bottom = height - 30;
    const laneH = (bottom - top) / layout.laneCount;
    // a square-root axis gives the first tokens, where most paths split, more room
    const xOf = (depth) => left + Math.sqrt(Math.min(1, depth / layout.maxDepth)) * (right - 40 - left);
    const yOf = (laneValue) => top + (laneValue + 0.5) * laneH;
    // answer nodes keep a minimum gap so their labels never overlap
    const gap = Math.min(26, (bottom - top) / Math.max(1, layout.answers.length));
    const ys = layout.answers.map((a) => yOf(a.lane));
    for (let k = 1; k < ys.length; k++) ys[k] = Math.max(ys[k], ys[k - 1] + gap);
    const overflow = ys.length ? ys[ys.length - 1] - bottom : 0;
    if (overflow > 0) for (let k = 0; k < ys.length; k++) ys[k] -= overflow;
    for (let k = ys.length - 2; k >= 0; k--) ys[k] = Math.min(ys[k], ys[k + 1] - gap);
    const answerAt = new Map(layout.answers.map((a, k) => [a.key, { x: right + 18, y: ys[k], answer: a }]));
    return { left, right, top, bottom, xOf, yOf, answerAt };
  }

  // ---------- drawing ----------

  function curve(x0, y0, x1, y1) {
    const mid = (x0 + x1) / 2;
    ctx.moveTo(x0, y0);
    ctx.bezierCurveTo(mid, y0, mid, y1, x1, y1);
  }

  /** Return the screen points along one sample path, for hit tests and highlights. */
  function routeOf(g, index) {
    const segs = layout.segments.filter((s) => s.leaves.includes(index)).sort((a, b) => a.from - b.from);
    return segs.map((s) => [g.xOf(s.from), g.yOf(s.fromLane), g.xOf(s.to), g.yOf(s.toLane), s]);
  }

  function drawRoute(route, upto) {
    for (const [x0, y0, x1, y1, s] of route) {
      if (s.from >= upto) break;
      const t = Math.min(1, (upto - s.from) / Math.max(1, s.to - s.from));
      if (t >= 1) curve(x0, y0, x1, y1);
      else {
        // a partly grown segment is drawn as a straight run toward its end
        ctx.moveTo(x0, y0);
        ctx.lineTo(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
      }
    }
  }

  function render(now) {
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    if (!layout || !width) return;
    const g = geometry();
    const grow = reducedMotion ? 1 : Math.min(1, (now - growStart) / GROW_MS);
    const upto = grow * layout.maxDepth;

    // the accepted set: a dashed disk around the correct answer node, as in step 1
    const correct = layout.answers.find((a) => a.correct);
    if (correct) {
      const node = g.answerAt.get(correct.key);
      ctx.beginPath();
      ctx.arc(node.x, node.y, 19, 0, 2 * Math.PI);
      ctx.fillStyle = colors.acceptFill;
      ctx.fill();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = colors.regionLine;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // segments: width grows with the number of paths, red shows the share of them that land
    for (const s of layout.segments) {
      if (s.from >= upto) continue;
      const route = [[g.xOf(s.from), g.yOf(s.fromLane), g.xOf(s.to), g.yOf(s.toLane), s]];
      const w = 1 + 1.4 * Math.sqrt(s.count);
      ctx.lineCap = "round";
      ctx.beginPath();
      drawRoute(route, upto);
      ctx.strokeStyle = colors.slate;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = w;
      ctx.stroke();
      if (s.share > 0) {
        ctx.beginPath();
        drawRoute(route, upto);
        ctx.strokeStyle = colors.red;
        ctx.globalAlpha = s.share;
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    // leaf ends and the dotted hand-off from each path to the answer it gives
    if (grow >= 1) {
      for (const s of layout.samples) {
        const route = routeOf(g, s.index);
        const last = route[route.length - 1];
        const node = g.answerAt.get(answerLabel(s));
        ctx.beginPath();
        curve(last[2], last[3], node.x - 12, node.y);
        ctx.setLineDash([2, 4]);
        ctx.strokeStyle = s.correct ? colors.red : colors.slate;
        ctx.globalAlpha = 0.45;
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
        ctx.beginPath();
        ctx.arc(last[2], last[3], 2.5, 0, 2 * Math.PI);
        ctx.fillStyle = s.correct ? colors.red : colors.slate;
        ctx.fill();
      }
    }

    // the selected or hovered path is drawn on top, thicker, with its what-if branch if shown
    const focus = hovered >= 0 ? hovered : selected;
    if (focus >= 0 && grow >= 1) {
      const route = routeOf(g, focus);
      ctx.beginPath();
      drawRoute(route, Infinity);
      ctx.strokeStyle = prompt.paths[focus].correct ? colors.red : colors.ink;
      ctx.lineWidth = 3;
      ctx.stroke();
      if (showWhatIf && focus === selected) drawWhatIf(g);
    }

    // root and labels
    const [rx, ry] = [g.xOf(0), g.yOf(layout.rootLane)];
    ctx.beginPath();
    ctx.arc(rx, ry, 5, 0, 2 * Math.PI);
    ctx.fillStyle = colors.ink;
    ctx.fill();
    ctx.font = "12px Instrument Sans, sans-serif";
    ctx.fillStyle = colors.muted;
    ctx.textAlign = "center";
    ctx.textAlign = "left";
    ctx.fillText("prompt x", rx - 4, g.bottom + 14);

    ctx.textAlign = "left";
    for (const [key, node] of g.answerAt) {
      const a = node.answer;
      const r = 3 + 2.2 * Math.sqrt(a.members.filter((i) => prompt.paths[i].kind === "sample").length);
      ctx.beginPath();
      ctx.arc(node.x - 6, node.y, Math.max(3, r), 0, 2 * Math.PI);
      ctx.fillStyle = a.correct ? colors.red : colors.slate;
      ctx.globalAlpha = grow >= 1 ? 1 : 0.35;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.fillStyle = a.correct ? colors.ink : colors.muted;
      ctx.font = `${a.correct ? 600 : 400} 12.5px Instrument Sans, sans-serif`;
      ctx.fillText(key === "none" ? "no answer" : key, node.x + 6, node.y + 4);
    }
    if (correct) {
      const node = g.answerAt.get(correct.key);
      ctx.font = "italic 13px Newsreader, serif";
      ctx.fillStyle = colors.muted;
      ctx.textAlign = "center";
      ctx.fillText("𝒮x", node.x - 6, node.y - 24);
    }
    ctx.font = "12px Instrument Sans, sans-serif";
    ctx.fillStyle = colors.muted;
    ctx.textAlign = "right";
    ctx.fillText("answers y(z)", width - 8, 16);
    ctx.textAlign = "left";
    ctx.textAlign = "center";
    ctx.fillText("tokens (square-root scale) →", (g.left + g.right) / 2, height - 8);

    if (grow < 1) raf = requestAnimationFrame(render);
    else raf = 0;
  }

  function drawWhatIf(g) {
    const what = prompt.paths.find((p) => p.kind === "whatif" && p.parent === selected);
    if (!what) return;
    // the fork sits on the selected path; the branch bends away from it toward its own answer
    const route = routeOf(g, selected);
    const seg = route.find(([, , , , s]) => s.from <= what.forkAt && what.forkAt < s.to) || route[route.length - 1];
    const [x0, y0, x1, y1, s] = seg;
    const fx = g.xOf(what.forkAt);
    const t = x1 > x0 ? Math.min(1, Math.max(0, (fx - x0) / (x1 - x0))) : 0;
    const fy = y0 + (y1 - y0) * (t * t * (3 - 2 * t));
    const ex = g.xOf(what.forkAt + what.tokens.length);
    const node = g.answerAt.get(answerLabel(what));
    const bend = fy + (fy < (g.top + g.bottom) / 2 ? 22 : -22);
    ctx.beginPath();
    curve(fx, fy, ex, bend);
    curve(ex, bend, node.x - 12, node.y);
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = what.correct ? colors.red : colors.teal;
    ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(fx, fy, 4.5, 0, 2 * Math.PI);
    ctx.fillStyle = colors.paper;
    ctx.fill();
    ctx.strokeStyle = colors.ink;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // ---------- text, readouts and the side column ----------

  function showPath() {
    const path = prompt.paths[selected];
    const what = prompt.paths.find((p) => p.kind === "whatif" && p.parent === selected);
    const box = $("#tk-text");
    box.textContent = "";
    const tokens = showWhatIf && what ? path.tokens.slice(0, what.forkAt) : path.tokens;
    const addToken = (t, k, extraClass) => {
      const span = document.createElement("span");
      span.className = `tok${extraClass ? ` ${extraClass}` : ""}`;
      span.textContent = t.t;
      // a less likely token is tinted more, so the reader sees where the path was uncertain
      span.style.setProperty("--doubt", (1 - t.p).toFixed(3));
      span.dataset.k = k;
      span.addEventListener("mouseenter", () => describeToken(t));
      span.addEventListener("focus", () => describeToken(t));
      span.tabIndex = 0;
      box.append(span);
      return span;
    };
    tokens.forEach((t, k) => {
      const span = addToken(t, k, what && k === what.forkAt ? "fork" : "");
      if (what && k === what.forkAt) span.title = "the least likely token on this path";
    });
    if (showWhatIf && what) what.tokens.forEach((t, k) => addToken(t, what.forkAt + k, "alt"));

    const shown = showWhatIf && what ? what : path;
    const all = showWhatIf && what ? path.tokens.slice(0, what.forkAt).concat(what.tokens) : path.tokens;
    $("#tk-path").innerHTML =
      `${all.length} tokens. π(z | x) = ${prob(pathProb(all))}. Answer: <b>${shown.answer === null ? "none" : answerLabel(shown)}</b>, ` +
      (shown.correct ? "<b class=\"lands\">lands in 𝒮<sub>x</sub></b>" : "misses");
    $("#tk-token").textContent = "Point at a token to see its probability and the model's other choices.";

    const button = $("#tk-whatif");
    button.hidden = !what;
    if (what) {
      const forced = what.tokens[0];
      const original = path.tokens[what.forkAt];
      // qwen writes " 3" as a space then "3", so a bare space is named with the token after it
      const withNext = (t, next) => (t.t.trim() || !next ? t.t : t.t + next.t);
      const forcedText = withNext(forced, what.tokens[1]);
      const originalText = withNext(original, path.tokens[what.forkAt + 1]);
      button.textContent = showWhatIf
        ? "Back to the sampled path"
        : `What if it wrote “${readable(forcedText)}” (${pct(forced.p)}) instead of “${readable(originalText)}” (${pct(original.p)})?`;
    }
  }

  function describeToken(t) {
    const alts = t.alts.map(([s, p]) => `“${readable(s)}” ${pct(p)}`).join(", ");
    $("#tk-token").innerHTML = `“${readable(t.t)}” had probability <b>${pct(t.p)}</b>. The model's other top choices: ${alts}.`;
  }

  function showSide() {
    const s = prompt.samples;
    const p = s.hits / s.n;
    const se = Math.sqrt((p * (1 - p)) / s.n);
    const treeHits = layout.samples.filter((x) => x.correct).length;
    const best = Math.max(...layout.samples.map((x) => pathProb(x.tokens)));
    $("#tk-question").textContent = prompt.question;
    $("#tk-p").textContent = pct(p);
    $("#tk-est").textContent = `${s.hits} / ${s.n} = ${pct(p)} ± ${pct(se)}`;
    $("#tk-tree").textContent = `${treeHits} / ${layout.samples.length}`;
    $("#tk-best").innerHTML = prob(best);

    // the answer histogram: the correct answer, then the most common wrong answers, then the rest
    const entries = Object.entries(s.answers).sort((a, b) => b[1] - a[1]);
    const right = String(prompt.answer);
    const top = entries.filter(([k]) => k !== "none").slice(0, 5);
    if (!top.some(([k]) => k === right) && s.answers[right]) top.push([right, s.answers[right]]);
    const shown = new Set(top.map(([k]) => k));
    const rest = entries.filter(([k]) => !shown.has(k) && k !== "none").reduce((n, [, c]) => n + c, 0);
    const rows = top.map(([k, c]) => ({ label: k, count: c, correct: k === right }));
    if (rest) rows.push({ label: "other numbers", count: rest });
    if (s.answers.none) rows.push({ label: "no answer", count: s.answers.none });
    const max = Math.max(...rows.map((r) => r.count));
    $("#tk-hist").innerHTML = rows.map((r) =>
      `<div class="hbar${r.correct ? " correct" : ""}"><span>${r.label}</span><i style="width:${(100 * r.count) / max}%"></i><b>${r.count}</b></div>`).join("");
    updateK();
  }

  // seeded so the vote readout does not flicker when the slider moves
  function seeded(seed) {
    let state = seed | 0;
    return () => {
      state = (state + 0x6d2b79f5) | 0;
      let t = Math.imul(state ^ (state >>> 15), 1 | state);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** Return how often a majority vote over k samples picks the correct answer; a sample with no answer abstains. */
  function majority(k) {
    const s = prompt.samples;
    const pool = Object.entries(s.answers).flatMap(([key, c]) => Array(c).fill(key));
    const uniform = seeded(11);
    const right = String(prompt.answer);
    let wins = 0;
    for (let trial = 0; trial < VOTE_TRIALS; trial++) {
      const votes = new Map();
      for (let j = 0; j < k; j++) {
        const a = pool[Math.floor(uniform() * pool.length)];
        if (a !== "none") votes.set(a, (votes.get(a) || 0) + 1);
      }
      if (!votes.size) continue;
      const most = Math.max(...votes.values());
      const leaders = [...votes].filter(([, c]) => c === most).map(([a]) => a);
      if (leaders[Math.floor(uniform() * leaders.length)] === right) wins++;
    }
    return wins / VOTE_TRIALS;
  }

  function updateK() {
    const k = +$("#tk-k").value;
    const p = prompt.samples.hits / prompt.samples.n;
    $("#tk-k-out").textContent = k;
    $("#tk-bok").textContent = pct(1 - (1 - p) ** k);
    const vote = majority(k);
    // a vote that never lost in the simulated trials is still not certain
    $("#tk-vote").textContent = vote === 1 ? ">99%" : pct(vote);
  }

  // ---------- interaction ----------

  function hitTest(x, y) {
    if (!layout) return -1;
    const g = geometry();
    let best = -1;
    let bestD = 9;
    for (const s of layout.samples) {
      for (const [x0, y0, x1, y1] of routeOf(g, s.index)) {
        for (let t = 0; t <= 1; t += 0.1) {
          // a point on the same cubic the renderer draws
          const e = t * t * (3 - 2 * t);
          const d = Math.hypot(x0 + (x1 - x0) * t - x, y0 + (y1 - y0) * e - y);
          if (d < bestD) { bestD = d; best = s.index; }
        }
      }
    }
    return best;
  }

  const local = (e) => {
    const r = canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };
  canvas.addEventListener("pointermove", (e) => {
    const h = hitTest(...local(e));
    if (h !== hovered) { hovered = h; canvas.style.cursor = h >= 0 ? "pointer" : ""; render(performance.now()); }
  });
  canvas.addEventListener("pointerleave", () => { hovered = -1; render(performance.now()); });
  canvas.addEventListener("click", (e) => {
    const h = hitTest(...local(e));
    if (h < 0) return;
    selected = h;
    showWhatIf = false;
    showPath();
    render(performance.now());
  });
  $("#tk-whatif").addEventListener("click", () => {
    showWhatIf = !showWhatIf;
    showPath();
    render(performance.now());
  });
  $("#tk-k").addEventListener("input", updateK);

  function choose(id) {
    prompt = data.prompts.find((p) => p.id === id);
    panel.querySelectorAll("[data-prompt]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.prompt === id)));
    layout = buildLayout();
    // start on the most likely sampled path, the one a reader would most often see
    selected = layout.samples.reduce((a, b) => (pathProb(b.tokens) > pathProb(a.tokens) ? b : a)).index;
    hovered = -1;
    showWhatIf = false;
    showPath();
    showSide();
    growStart = performance.now();
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(render);
  }

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    width = canvas.clientWidth;
    height = canvas.clientHeight;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    render(performance.now());
  }

  canvas.style.aspectRatio = String(ASPECT);
  readColors();
  new ResizeObserver(resize).observe(canvas);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readColors(); render(performance.now()); });
  new MutationObserver(() => { readColors(); render(performance.now()); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  fetch("data/token-trees.json?v=2")
    .then((r) => r.json())
    .catch((err) => {
      $("#tk-path").textContent = "The token data did not load. Serve the page over http, not from a file.";
      throw err;
    })
    .then((json) => {
      data = json;
      $("#tk-model").textContent = `${json.model.split("/").pop()} at temperature ${json.temperature}`;
      panel.querySelectorAll("[data-prompt]").forEach((b) => b.addEventListener("click", () => choose(b.dataset.prompt)));
      choose(panel.querySelector("[data-prompt]").dataset.prompt);
    });
})();
