// pure math for steps 11 and 12: what rl can add (pass@k coverage, process vs outcome reward) and how
// the common rl algorithms differ (reinforce, rloo, grpo, dr. grpo, ppo-clip). no dom, so tools/check-beyond.js runs it.
(function (root) {
  "use strict";
  const W = typeof module !== "undefined" && module.exports ? require("./walks-model.js") : root.ReasoningWalks;

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
  const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
  const CLOUD = (() => {
    const n = gaussianFrom(seededUniform(3));
    return Array.from({ length: 2000 }, () => [n(), n()]);
  })();
  function diskHit(m, sd, center, r) {
    let h = 0;
    for (const [gx, gy] of CLOUD) if (Math.hypot(m[0] + sd * gx - center[0], m[1] + sd * gy - center[1]) <= r) h++;
    return h / CLOUD.length;
  }

  function adam(n, lr) {
    const m = new Array(n).fill(0);
    const v = new Array(n).fill(0);
    let t = 0;
    return (params, grad) => {
      t++;
      return params.map((p, i) => {
        m[i] = 0.9 * m[i] + 0.1 * grad[i];
        v[i] = 0.999 * v[i] + 0.001 * grad[i] ** 2;
        return p + (lr * (m[i] / (1 - 0.9 ** t))) / (Math.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8);
      });
    };
  }

  // ---------- coverage: one shared policy, many tasks ----------

  const ARC = W.arcFor(W.STEPS);
  // targets need different vertical corrections in no order a constant offset could exploit
  const NEEDS = [0.05, -0.3, 0.22, -0.08, 0.34, -0.2, 0.12, -0.36, 0.28, -0.02];
  const COVERAGE = {
    radius: 0.12,
    baseSigma: 0.2,
    tasks: NEEDS.map((v, i) => {
      const start = [-0.9, -0.9 + (i * 1.8) / (NEEDS.length - 1)];
      return { id: i, start, target: [start[0] + ARC[0] + 0.2, start[1] + ARC[1] + v] };
    }),
  };

  function coverageMean(policy, task) {
    return [task.start[0] + ARC[0] + policy.theta[0], task.start[1] + ARC[1] + policy.theta[1]];
  }
  function coverageHit(policy, task) {
    return diskHit(coverageMean(policy, task), Math.exp(policy.logSigma), task.target, COVERAGE.radius);
  }
  /** Return the mean over tasks of pass@k for each k, the chance that at least one of k samples lands. */
  function passAtK(policy, ks) {
    const ps = COVERAGE.tasks.map((t) => coverageHit(policy, t));
    return ks.map((k) => mean(ps.map((p) => 1 - (1 - p) ** k)));
  }

  /** Train the shared policy with a 0/1 reward and a group-mean baseline; return snapshots per step. */
  function trainCoverage({ steps = 300, group = 8, lr = 0.02, beta = 0, seed = 1 } = {}) {
    const normal = gaussianFrom(seededUniform(seed));
    let params = [0, 0, Math.log(COVERAGE.baseSigma)];
    const update = adam(3, lr);
    const history = [];
    for (let s = 0; s < steps; s++) {
      const grad = [0, 0, 0];
      const policy = { theta: [params[0], params[1]], logSigma: params[2] };
      const sig = Math.exp(params[2]);
      for (const task of COVERAGE.tasks) {
        const m = coverageMean(policy, task);
        const ys = Array.from({ length: group }, () => [m[0] + sig * normal(), m[1] + sig * normal()]);
        const r = ys.map((y) => (Math.hypot(y[0] - task.target[0], y[1] - task.target[1]) <= COVERAGE.radius ? 1 : 0));
        const b = mean(r);
        ys.forEach((y, i) => {
          const d = [y[0] - m[0], y[1] - m[1]];
          const a = r[i] - b;
          grad[0] += (a * d[0]) / sig ** 2;
          grad[1] += (a * d[1]) / sig ** 2;
          grad[2] += a * ((d[0] ** 2 + d[1] ** 2) / sig ** 2 - 2);
        });
      }
      // an optional pull back to the base spread, the kl term's effect on sigma
      grad[2] -= beta * COVERAGE.tasks.length * group * (params[2] - Math.log(COVERAGE.baseSigma));
      params = update(params, grad.map((g) => g / (COVERAGE.tasks.length * group)));
      params[2] = Math.max(Math.log(0.02), Math.min(Math.log(0.6), params[2]));
      history.push({ theta: [params[0], params[1]], logSigma: params[2] });
    }
    return history;
  }

  // ---------- process vs outcome reward ----------

  const PROCESS = {
    start: [-0.75, 0],
    sigma: 0.2,
    target: [0.78, 0.38],
    radius: 0.08,
    checkpoints: [8, 16, 24, 32, 40],
  };
  const PROCESS_THETA = [PROCESS.target[0] - PROCESS.start[0] - ARC[0], PROCESS.target[1] - PROCESS.start[1] - ARC[1]];
  // the corridor a good chain of thought stays in: wide early, as tight as the target at the end
  const corridor = (k) => PROCESS.radius + 0.14 * (1 - k / W.STEPS);
  /** Return where a correct path passes after k steps. */
  function idealPoint(k) {
    const a = W.arcFor(k);
    const t = k * W.DT;
    return [PROCESS.start[0] + a[0] + PROCESS_THETA[0] * t, PROCESS.start[1] + a[1] + PROCESS_THETA[1] * t];
  }

  function processPath(policy, normal) {
    const s = Math.exp(policy.logSigma) * Math.sqrt(W.DT);
    const pts = [PROCESS.start];
    let sq = 0;
    for (let k = 0; k < W.STEPS; k++) {
      const d = W.driftAt(k);
      const e = [s * normal(), s * normal()];
      sq += (e[0] ** 2 + e[1] ** 2) / s ** 2;
      const p = pts[k];
      pts.push([p[0] + d[0] + policy.theta[0] * W.DT + e[0], p[1] + d[1] + policy.theta[1] * W.DT + e[1]]);
    }
    return { pts, sq };
  }
  const outcomeOf = (pts) => (Math.hypot(pts[W.STEPS][0] - PROCESS.target[0], pts[W.STEPS][1] - PROCESS.target[1]) <= PROCESS.radius ? 1 : 0);
  /** Return the fraction of checkpoints where the path stays inside the corridor. */
  function processScore(pts) {
    return mean(PROCESS.checkpoints.map((k) => {
      const q = idealPoint(k);
      return Math.hypot(pts[k][0] - q[0], pts[k][1] - q[1]) <= corridor(k) ? 1 : 0;
    }));
  }

  /** Train with outcome reward (0/1 at the end) or process reward (credit per checkpoint); return history. */
  function trainProcess({ reward = "outcome", steps = 300, batch = 64, lr = 0.02, beta = 0.3, seed = 1 } = {}) {
    const normal = gaussianFrom(seededUniform(seed));
    let params = [0, 0, Math.log(PROCESS.sigma)];
    const update = adam(3, lr);
    const evalNormal = () => gaussianFrom(seededUniform(77));
    const history = [];
    const height = 3;
    for (let s = 0; s <= steps; s++) {
      const policy = { theta: [params[0], params[1]], logSigma: params[2] };
      if (s % 10 === 0 || s === steps) {
        const n = evalNormal();
        let pass = 0;
        for (let i = 0; i < 400; i++) pass += outcomeOf(processPath(policy, n).pts);
        history.push({ step: s, pass: pass / 400, theta: policy.theta.slice(), sigma: Math.exp(params[2]) });
      }
      if (s === steps) break;
      const sig = Math.exp(params[2]);
      const m = [PROCESS.start[0] + ARC[0] + params[0], PROCESS.start[1] + ARC[1] + params[1]];
      const samples = Array.from({ length: batch }, () => processPath(policy, normal));
      // the kl to the base model has a closed form for this gaussian family
      const R = samples.map(({ pts }) => height * (reward === "process" ? processScore(pts) : outcomeOf(pts)));
      const b = mean(R);
      const grad = [0, 0, 0];
      samples.forEach(({ pts, sq }, i) => {
        const y = pts[W.STEPS];
        const a = R[i] - b;
        grad[0] += (a * (y[0] - m[0])) / sig ** 2;
        grad[1] += (a * (y[1] - m[1])) / sig ** 2;
        grad[2] += a * (sq - 2 * W.STEPS);
      });
      const r2 = PROCESS.sigma ** 2;
      grad[0] = grad[0] / batch - (beta * params[0]) / r2;
      grad[1] = grad[1] / batch - (beta * params[1]) / r2;
      grad[2] = grad[2] / batch - beta * (2 * (sig ** 2 / r2) - 2);
      params = update(params, grad);
      params[2] = Math.max(Math.log(0.02), Math.min(Math.log(0.6), params[2]));
    }
    return history;
  }

  // ---------- rl algorithms, with a policy that also chooses its chain length ----------

  const LENGTH = {
    start: [-0.75, 0],
    target: [0.5, -0.26],
    radius: 0.2,
    sigma40: Math.sqrt(2 * 0.008),
    lengthSd: 4,
    minL: 12,
    maxL: 90,
  };
  const ALGORITHMS = ["reinforce", "rloo", "grpo", "drgrpo", "ppo"];

  function lengthPolicy(params) {
    return { theta: [params[0], params[1]], logSigma: params[2], muL: params[3] };
  }
  function lengthSample(params, normal) {
    const ell = params[3] + LENGTH.lengthSd * normal();
    const L = Math.max(LENGTH.minL, Math.min(LENGTH.maxL, Math.round(ell)));
    const a = W.arcFor(L);
    const t = L / W.STEPS;
    const sd = Math.exp(params[2]) * Math.sqrt(t);
    const m = [LENGTH.start[0] + a[0] + params[0] * t, LENGTH.start[1] + a[1] + params[1] * t];
    const y = [m[0] + sd * normal(), m[1] + sd * normal()];
    const hit = Math.hypot(y[0] - LENGTH.target[0], y[1] - LENGTH.target[1]) <= LENGTH.radius ? 1 : 0;
    return { ell, L, y, hit };
  }
  // log-probability and its gradient for one sample under params, without constants
  function logProbGrad(params, s) {
    const t = s.L / W.STEPS;
    const a = W.arcFor(s.L);
    const sig = Math.exp(params[2]);
    const v = sig ** 2 * t;
    const m = [LENGTH.start[0] + a[0] + params[0] * t, LENGTH.start[1] + a[1] + params[1] * t];
    const d = [s.y[0] - m[0], s.y[1] - m[1]];
    const dl = s.ell - params[3];
    const lp = -(d[0] ** 2 + d[1] ** 2) / (2 * v) - 2 * params[2] - dl ** 2 / (2 * LENGTH.lengthSd ** 2);
    const g = [(d[0] * t) / v, (d[1] * t) / v, (d[0] ** 2 + d[1] ** 2) / v - 2, dl / LENGTH.lengthSd ** 2];
    return { lp, g };
  }

  /**
   * Train one algorithm on the verifiable task with a learnable length; return per-step stats:
   * success rate, and the mean length of correct and of incorrect answers.
   */
  function trainLength({ algorithm = "grpo", steps = 300, group = 16, groups = 4, lr = 0.02, seed = 1, epochs = 4, clip = 0.2 } = {}) {
    const normal = gaussianFrom(seededUniform(seed));
    let params = [0, 0, Math.log(LENGTH.sigma40), W.STEPS];
    const update = adam(4, lr);
    const history = [];
    for (let s = 0; s < steps; s++) {
      const batch = [];
      for (let q = 0; q < groups; q++) {
        const g = Array.from({ length: group }, () => lengthSample(params, normal));
        const r = g.map((x) => x.hit);
        const mu = mean(r);
        const sd = Math.sqrt(mean(r.map((x) => (x - mu) ** 2)));
        g.forEach((x, i) => {
          let adv;
          let weight = 1;
          if (algorithm === "rloo") adv = r[i] - (mu * group - r[i]) / (group - 1);
          else if (algorithm === "grpo") {
            adv = sd > 0 ? (r[i] - mu) / sd : 0;
            // grpo averages each response's loss over its own tokens
            weight = 1 / x.L;
          } else if (algorithm === "drgrpo") {
            adv = r[i] - mu;
            weight = 1 / LENGTH.maxL;
          } else adv = r[i] - mu;
          batch.push({ ...x, adv: adv * weight });
        });
      }
      const correct = batch.filter((x) => x.hit);
      const wrong = batch.filter((x) => !x.hit);
      history.push({
        step: s,
        success: correct.length / batch.length,
        lenCorrect: correct.length ? mean(correct.map((x) => x.L)) : NaN,
        lenWrong: wrong.length ? mean(wrong.map((x) => x.L)) : NaN,
        muL: params[3],
      });
      if (algorithm === "ppo") {
        const old = batch.map((x) => logProbGrad(params, x).lp);
        for (let e = 0; e < epochs; e++) {
          const grad = [0, 0, 0, 0];
          batch.forEach((x, i) => {
            const { lp, g } = logProbGrad(params, x);
            const ratio = Math.exp(lp - old[i]);
            // the clipped surrogate stops pushing once the ratio leaves [1 - clip, 1 + clip] in the advantage's direction
            if ((x.adv > 0 && ratio > 1 + clip) || (x.adv < 0 && ratio < 1 - clip)) return;
            for (let j = 0; j < 4; j++) grad[j] += (x.adv * ratio * g[j]) / batch.length;
          });
          params = update(params, grad);
        }
      } else {
        const grad = [0, 0, 0, 0];
        batch.forEach((x) => {
          const { g } = logProbGrad(params, x);
          for (let j = 0; j < 4; j++) grad[j] += (x.adv * g[j]) / batch.length;
        });
        params = update(params, grad);
      }
      params[2] = Math.max(Math.log(0.02), Math.min(Math.log(0.6), params[2]));
      params[3] = Math.max(LENGTH.minL, Math.min(LENGTH.maxL, params[3]));
    }
    return { history, params };
  }

  const api = {
    COVERAGE, PROCESS, LENGTH, ALGORITHMS, seededUniform, gaussianFrom,
    coverageMean, coverageHit, passAtK, trainCoverage,
    idealPoint, corridor, processPath, processScore, outcomeOf, trainProcess,
    lengthSample, trainLength,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Beyond = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
