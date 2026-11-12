// pure math for the gepa steps: a frozen model whose behaviour is steered only by a two-module prompt,
// a simulated reflector that turns text feedback into prompt edits, pareto selection, merge, and a grpo
// baseline that trains the same four numbers as weights. no dom, so tools/check-gepa.js can run it.
(function (root) {
  "use strict";
  const W = typeof module !== "undefined" && module.exports ? require("./walks-model.js") : root.ReasoningWalks;

  const DIFFUSION = 0.003;
  const SD = Math.sqrt(2 * DIFFUSION);
  const RADIUS = 0.15;
  const VIEW = [-1.2, 1.4, -1.5, 1.3];
  const ARC = W.DRIFT.reduce((m, d) => [m[0] + d[0], m[1] + d[1]], [0, 0]);
  // what each module would have to say, written as shift + tilt * (task height)
  // the targets fan out with task height (v tilt above zero), so no two targets overlap
  const NEEDED = { H: { shift: 0.33, tilt: 0.15 }, V: { shift: -0.1, tilt: 0.3 } };
  const TASKS = Array.from({ length: 8 }, (_, i) => {
    const y = -0.9 + (i * 1.8) / 7;
    const start = [-0.9, y];
    const target = [
      start[0] + ARC[0] + NEEDED.H.shift + NEEDED.H.tilt * y,
      start[1] + ARC[1] + NEEDED.V.shift + NEEDED.V.tilt * y,
    ];
    return { id: i, start, target, height: y };
  });
  const MODULES = ["H", "V"];
  const AXIS = { H: 0, V: 1 };

  function seededUniform(seed) {
    let state = seed | 0;
    return () => {
      state = (state + 0x6d2b79f5) | 0;
      let t = Math.imul(state ^ (state >>> 15), 1 | state);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const gaussianFrom = W.gaussianFrom;

  // ---------- prompts ----------

  let lineIds = 0;
  function amount(m) {
    const a = Math.abs(m);
    return a < 0.08 ? "slightly" : a < 0.2 ? "somewhat" : "much";
  }
  /** Return the instruction text for one prompt line. */
  function lineText(module, kind, m) {
    if (module === "H" && kind === "shift") return `End ${amount(m)} further ${m > 0 ? "right" : "left"} on every task.`;
    if (module === "H") return `For higher tasks, end ${amount(m)} further ${m > 0 ? "right" : "left"}; for lower tasks, the opposite.`;
    if (kind === "shift") return `End ${amount(m)} ${m > 0 ? "higher" : "lower"} on every task.`;
    return `For higher tasks, end ${amount(m)} ${m > 0 ? "higher" : "lower"}; for lower tasks, the opposite.`;
  }
  function makeLine(module, kind, m) {
    return { id: ++lineIds, kind, m, text: lineText(module, kind, m) };
  }

  const BASE_TEXT = { H: "Read the task and give a final answer.", V: "Think step by step." };
  let moduleIds = 0;
  function makeModule(name, lines) {
    return { id: ++moduleIds, name, lines };
  }
  function baseCandidate() {
    return { id: 0, parent: null, parents: [], modules: { H: makeModule("H", []), V: makeModule("V", []) }, next: 0, origin: "base" };
  }

  function promptOffset(cand, task) {
    const o = [0, 0];
    for (const name of MODULES) {
      for (const l of cand.modules[name].lines) o[AXIS[name]] += l.kind === "shift" ? l.m : l.m * task.height;
    }
    return o;
  }

  // ---------- rollouts ----------

  function endpointMean(offset, task) {
    return [task.start[0] + ARC[0] + offset[0], task.start[1] + ARC[1] + offset[1]];
  }

  /** Run one rollout and return {hit, miss, end}: miss is the endpoint minus the target center. */
  function rollout(offset, task, normal) {
    const m = endpointMean(offset, task);
    const end = [m[0] + SD * normal(), m[1] + SD * normal()];
    const miss = [end[0] - task.target[0], end[1] - task.target[1]];
    return { task: task.id, end, miss, hit: Math.hypot(miss[0], miss[1]) <= RADIUS };
  }

  /** Return the text a feedback function would write about one rollout. */
  function feedbackText(r) {
    if (r.hit) return "Correct.";
    const parts = [];
    if (Math.abs(r.miss[0]) > 0.03) parts.push(`${Math.abs(r.miss[0]).toFixed(2)} too far ${r.miss[0] > 0 ? "right" : "left"}`);
    if (Math.abs(r.miss[1]) > 0.03) parts.push(`${Math.abs(r.miss[1]).toFixed(2)} too ${r.miss[1] > 0 ? "high" : "low"}`);
    return `Wrong. Ended ${parts.join(" and ")}.`;
  }

  function walkPoints(offset, task, normal) {
    const sd = Math.sqrt(2 * DIFFUSION * W.DT);
    const pts = [task.start];
    for (let k = 0; k < W.STEPS; k++) {
      const p = pts[k];
      pts.push([p[0] + W.DRIFT[k][0] + offset[0] * W.DT + sd * normal(), p[1] + W.DRIFT[k][1] + offset[1] * W.DT + sd * normal()]);
    }
    return pts;
  }

  // a fixed noise cloud, so test success does not flicker and costs no rollouts
  const CLOUD = (() => {
    const n = gaussianFrom(seededUniform(3));
    return Array.from({ length: 1500 }, () => [n(), n()]);
  })();
  function hitProbability(offset, task) {
    const m = endpointMean(offset, task);
    let hits = 0;
    for (const [gx, gy] of CLOUD) if (Math.hypot(m[0] + SD * gx - task.target[0], m[1] + SD * gy - task.target[1]) <= RADIUS) hits++;
    return hits / CLOUD.length;
  }
  /** Return the mean hit probability over all tasks for an offset function. */
  function testSuccess(offsetOf) {
    return TASKS.reduce((s, t) => s + hitProbability(offsetOf(t), t), 0) / TASKS.length;
  }

  // ---------- reflection ----------

  /**
   * Return a child module: the reflector reads the traces' feedback for this module's axis and writes or
   * edits one line. Without reflection it edits a random line by a random amount.
   */
  // reflector quality: at Q0 the reflector behaves as before; above it its sense of amount sharpens,
  // below it a growing share of its edits are random
  const Q0 = 0.7;

  function reflect(module, traces, uniform, normal, reflective, quality = Q0, feedback = "text") {
    let kind;
    let delta;
    const pRandom = quality < Q0 ? (Q0 - quality) / Q0 : 0;
    // only draw the extra random number when it matters, so default runs stay identical
    const randomEdit = !reflective || (pRandom > 0 && uniform() < pRandom);
    if (!randomEdit && feedback === "binary") {
      // pass or fail only: the reflector knows how often this module failed, not in which direction
      const failRate = traces.filter((r) => !r.hit).length / traces.length;
      kind = uniform() < 0.5 ? "shift" : "tilt";
      delta = (uniform() < 0.5 ? -1 : 1) * 0.3 * failRate * (0.5 + uniform());
    } else if (!randomEdit) {
      const axis = AXIS[module.name];
      const hs = traces.map((r) => TASKS[r.task].height);
      const es = traces.map((r) => r.miss[axis]);
      const hm = hs.reduce((a, b) => a + b, 0) / hs.length;
      const em = es.reduce((a, b) => a + b, 0) / es.length;
      const varH = hs.reduce((a, h) => a + (h - hm) ** 2, 0);
      const slope = varH > 1e-6 ? hs.reduce((a, h, i) => a + (h - hm) * (es[i] - em), 0) / varH : 0;
      const sdH = Math.sqrt(varH / hs.length);
      // the reflector names the bigger pattern: a miss that grows with task height, or one shared by all tasks
      kind = Math.abs(slope) * sdH > Math.abs(em) ? "tilt" : "shift";
      const correction = kind === "tilt" ? -slope : -em;
      // an llm's sense of "how much" is rough; at Q0 this is 0.5 + 0.7u
      const sharp = Math.max(0, (quality - Q0) / (1 - Q0));
      const center = 0.85 + 0.15 * sharp;
      const halfWidth = quality >= Q0 ? 0.35 * (1 - sharp) : 0.35;
      delta = correction * (center + halfWidth * (2 * uniform() - 1));
    } else {
      kind = uniform() < 0.5 ? "shift" : "tilt";
      delta = 0.25 * normal();
    }
    delta = Math.round(delta / 0.01) * 0.01;
    const lines = module.lines.map((l) => ({ ...l }));
    const i = lines.findIndex((l) => l.kind === kind);
    if (i >= 0) lines[i] = makeLine(module.name, kind, Math.round((lines[i].m + delta) * 100) / 100);
    else lines.push(makeLine(module.name, kind, delta));
    return { module: makeModule(module.name, lines.filter((l) => Math.abs(l.m) >= 0.005)), kind, delta };
  }

  // ---------- pareto selection ----------

  /** Return the pareto candidates and how many tasks each is best on (algorithm 2 of the paper). */
  function paretoFront(pool, scores) {
    const front = new Map();
    for (let t = 0; t < TASKS.length; t++) {
      const best = Math.max(...pool.map((c) => scores.get(c.id)[t]));
      for (const c of pool) if (scores.get(c.id)[t] === best) front.set(c.id, (front.get(c.id) || 0) + 1);
    }
    const ids = [...front.keys()];
    const dominated = (a, b) => {
      const sa = scores.get(a);
      const sb = scores.get(b);
      return sb.every((x, t) => x >= sa[t]) && sb.some((x, t) => x > sa[t]);
    };
    for (const a of ids) if (ids.some((b) => b !== a && front.has(b) && dominated(a, b))) front.delete(a);
    return front;
  }

  const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;

  // ---------- gepa ----------

  const MERGE_EVERY = 4;
  // the paper invokes merge at most five times per run
  const MAX_MERGES = 5;

  /**
   * Create a GEPA run. options: seed, selection ("pareto" | "greedy"), reflective, merge.
   * iterate() runs one proposal and returns an event describing it.
   */
  function createGepa({
    seed = 1, selection = "pareto", reflective = true, merge = false,
    quality = Q0, feedback = "text", minibatch: MINIBATCH = 3, paretoSamples: PARETO_SAMPLES = 4,
  } = {}) {
    const uniform = seededUniform(seed);
    const normal = gaussianFrom(uniform);
    const pool = [baseCandidate()];
    // ids are per run, so runs that interleave on one page never share or reuse an id
    let candidateIds = 1;
    const byId = new Map([[0, pool[0]]]);
    const scores = new Map();
    let rollouts = 0;
    let iteration = 0;
    let merges = 0;
    const history = [];

    const run = (cand, task) => {
      rollouts++;
      return rollout(promptOffset(cand, task), task, normal);
    };
    const evaluatePareto = (cand) => {
      scores.set(cand.id, TASKS.map((t) => {
        let h = 0;
        for (let k = 0; k < PARETO_SAMPLES; k++) h += run(cand, t).hit ? 1 : 0;
        return h / PARETO_SAMPLES;
      }));
    };
    const best = () => pool.reduce((a, b) => (mean(scores.get(b.id)) > mean(scores.get(a.id)) ? b : a));
    const record = () => history.push({ rollouts, test: testSuccess((t) => promptOffset(best(), t)), best: best().id });
    const minibatch = () => {
      const ids = TASKS.map((t) => t.id);
      for (let i = ids.length - 1; i > 0; i--) {
        const j = Math.floor(uniform() * (i + 1));
        [ids[i], ids[j]] = [ids[j], ids[i]];
      }
      return ids.slice(0, MINIBATCH).map((i) => TASKS[i]);
    };
    const select = () => {
      if (selection === "greedy") return best();
      const front = paretoFront(pool, scores);
      const total = [...front.values()].reduce((a, b) => a + b, 0);
      let u = uniform() * total;
      for (const [id, f] of front) {
        u -= f;
        if (u <= 0) return byId.get(id);
      }
      return byId.get([...front.keys()].pop());
    };
    const ancestors = (c) => {
      const out = [];
      for (let p = c; p; p = p.parent === null ? null : byId.get(p.parent)) out.push(p.id);
      return out;
    };
    const add = (child) => {
      pool.push(child);
      byId.set(child.id, child);
      evaluatePareto(child);
    };

    evaluatePareto(pool[0]);
    record();

    function tryMerge() {
      const front = [...paretoFront(pool, scores).keys()].map((id) => byId.get(id));
      if (front.length < 2) return null;
      const i = front[Math.floor(uniform() * front.length)];
      const others = front.filter((c) => c !== i);
      const j = others[Math.floor(uniform() * others.length)];
      const ai = ancestors(i);
      const common = ancestors(j).find((id) => ai.includes(id));
      if (common === undefined || common === i.id || common === j.id) return null;
      const anc = byId.get(common);
      const modules = {};
      for (const name of MODULES) {
        const ci = i.modules[name].id !== anc.modules[name].id;
        const cj = j.modules[name].id !== anc.modules[name].id;
        if (ci && !cj) modules[name] = i.modules[name];
        else if (cj && !ci) modules[name] = j.modules[name];
        else if (ci && cj) modules[name] = mean(scores.get(i.id)) >= mean(scores.get(j.id)) ? i.modules[name] : j.modules[name];
        else modules[name] = anc.modules[name];
      }
      // a merge must take something from each side, or it is just a copy
      if (MODULES.every((n) => modules[n].id === i.modules[n].id) || MODULES.every((n) => modules[n].id === j.modules[n].id)) return null;
      const child = { id: candidateIds++, parent: i.id, parents: [i.id, j.id], modules, next: 0, origin: "merge" };
      const batch = minibatch();
      const si = batch.map((t) => (run(i, t).hit ? 1 : 0)).reduce((a, b) => a + b, 0);
      const sj = batch.map((t) => (run(j, t).hit ? 1 : 0)).reduce((a, b) => a + b, 0);
      const childTraces = batch.map((t) => run(child, t));
      const sc = childTraces.filter((r) => r.hit).length;
      const accepted = sc >= Math.max(si, sj);
      if (accepted) add(child);
      return { type: "merge", parents: [i.id, j.id], child, batch: batch.map((t) => t.id), childTraces, scoreBefore: Math.max(si, sj), scoreAfter: sc, accepted };
    }

    function iterate() {
      iteration++;
      let event = null;
      if (merge && merges < MAX_MERGES && iteration % MERGE_EVERY === 0) {
        event = tryMerge();
        if (event && event.accepted) merges++;
      }
      if (!event) {
        const parent = select();
        const name = MODULES[parent.next % MODULES.length];
        parent.next++;
        const batch = minibatch();
        const parentTraces = batch.map((t) => run(parent, t));
        const { module, kind, delta } = reflect(parent.modules[name], parentTraces, uniform, normal, reflective, quality, feedback);
        const child = { id: candidateIds++, parent: parent.id, parents: [parent.id], modules: { ...parent.modules, [name]: module }, next: 0, origin: reflective ? "reflection" : "random" };
        const childTraces = batch.map((t) => run(child, t));
        const before = parentTraces.filter((r) => r.hit).length;
        const after = childTraces.filter((r) => r.hit).length;
        const accepted = after > before;
        if (accepted) add(child);
        else candidateIds--;
        event = { type: "mutate", parent: parent.id, module: name, kind, delta, child, batch: batch.map((t) => t.id), parentTraces, childTraces, scoreBefore: before, scoreAfter: after, accepted };
      }
      record();
      return event;
    }

    return {
      iterate,
      get pool() { return pool; },
      get scores() { return scores; },
      get rollouts() { return rollouts; },
      get history() { return history; },
      best,
      front: () => paretoFront(pool, scores),
      byId: (id) => byId.get(id),
    };
  }

  /** Run GEPA until the rollout budget is spent and return its history. */
  function runGepa(options, budget) {
    const g = createGepa(options);
    while (g.rollouts < budget) g.iterate();
    return g;
  }

  // ---------- grpo on weights ----------

  /** Run GRPO on the four offset weights with a 0/1 reward and return its history. */
  function runGrpo({ seed = 1, budget = 24000, group = 8, tasksPerStep = 4, lr = 0.01 } = {}) {
    const normal = gaussianFrom(seededUniform(seed));
    // theta = [h shift, h tilt, v shift, v tilt], the same capacity as the two prompt modules
    let theta = [0, 0, 0, 0];
    let m = [0, 0, 0, 0];
    let v = [0, 0, 0, 0];
    let t = 0;
    let rollouts = 0;
    let cursor = 0;
    const offsetOf = (task) => [theta[0] + theta[1] * task.height, theta[2] + theta[3] * task.height];
    const history = [{ rollouts: 0, test: testSuccess(offsetOf) }];
    while (rollouts < budget) {
      const grad = [0, 0, 0, 0];
      let n = 0;
      for (let k = 0; k < tasksPerStep; k++) {
        const task = TASKS[cursor++ % TASKS.length];
        const mu = endpointMean(offsetOf(task), task);
        const rs = Array.from({ length: group }, () => rollout(offsetOf(task), task, normal));
        rollouts += group;
        const r = rs.map((x) => (x.hit ? 1 : 0));
        const rm = mean(r);
        const rsd = Math.sqrt(mean(r.map((x) => (x - rm) ** 2)));
        // all-pass or all-fail groups carry no signal, which is why sparse rewards learn slowly
        if (rsd === 0) continue;
        rs.forEach((x, i) => {
          const a = (r[i] - rm) / rsd;
          const gx = (x.end[0] - mu[0]) / SD ** 2;
          const gy = (x.end[1] - mu[1]) / SD ** 2;
          grad[0] += a * gx;
          grad[1] += a * gx * task.height;
          grad[2] += a * gy;
          grad[3] += a * gy * task.height;
          n++;
        });
      }
      if (n) {
        t++;
        for (let j = 0; j < 4; j++) {
          const g = grad[j] / n;
          m[j] = 0.9 * m[j] + 0.1 * g;
          v[j] = 0.999 * v[j] + 0.001 * g * g;
          theta[j] += (lr * (m[j] / (1 - 0.9 ** t))) / (Math.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8);
        }
      }
      if (history[history.length - 1].rollouts + 96 <= rollouts) history.push({ rollouts, test: testSuccess(offsetOf) });
    }
    history.push({ rollouts, test: testSuccess(offsetOf) });
    return { history, theta };
  }

  /** Return the rollouts a history needs to first reach a test success, or Infinity. */
  function rolloutsToReach(history, level) {
    const h = history.find((x) => x.test >= level);
    return h ? h.rollouts : Infinity;
  }

  const api = {
    TASKS, VIEW, RADIUS, NEEDED, MODULES, BASE_TEXT, ARC, Q0,
    seededUniform, gaussianFrom, promptOffset, rollout, feedbackText, walkPoints, hitProbability, testSuccess,
    paretoFront, createGepa, runGepa, runGrpo, rolloutsToReach, lineText,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.GEPA = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
