// pure math for the rlhf tutorial: hidden utility, preference labels, bradley-terry reward model,
// reinforce with a kl penalty, and dpo. the policy is the part 1 walk with a learnable end offset and spread,
// so its endpoint is y ~ N(refMean + theta, sigma^2 I) and every kl and log-prob has a closed form.
(function (root) {
  "use strict";
  const W = typeof module !== "undefined" && module.exports ? require("./walks-model.js") : root.ReasoningWalks;

  const START = [-0.75, 0];
  const VIEW = [-1.0, 1.55, -0.85, 0.95];
  const REF_SIGMA = 0.2;
  const REF_MEAN = W.DRIFT.reduce((m, d) => [m[0] + d[0], m[1] + d[1]], START.slice());

  // the hidden human utility: a bump that sits partly outside where the base model goes
  const UTILITY = { center: [0.78, 0.38], width: 0.26, height: 3 };
  // the verifier for the rlvr comparison accepts a disk around the same best answer
  const VERIFIER = { center: UTILITY.center, r: 0.16 };

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
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const dist2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;

  function trueUtility(y) {
    return UTILITY.height * Math.exp(-dist2(y, UTILITY.center) / (2 * UTILITY.width ** 2));
  }

  function verifierReward(y) {
    return dist2(y, VERIFIER.center) <= VERIFIER.r ** 2 ? 1 : 0;
  }

  // ---------- policy ----------

  function refPolicy() {
    return { theta: [0, 0], logSigma: Math.log(REF_SIGMA) };
  }

  function policyMean(policy) {
    return [REF_MEAN[0] + policy.theta[0], REF_MEAN[1] + policy.theta[1]];
  }

  /** Return log pi(y) for an isotropic 2D gaussian policy, without the constant. */
  function logProb(policy, y) {
    const s2 = Math.exp(2 * policy.logSigma);
    return -dist2(y, policyMean(policy)) / (2 * s2) - 2 * policy.logSigma;
  }

  /** Return KL(policy || reference) in closed form. */
  function klToRef(policy) {
    const s2 = Math.exp(2 * policy.logSigma);
    const r2 = REF_SIGMA ** 2;
    return s2 / r2 + (policy.theta[0] ** 2 + policy.theta[1] ** 2) / (2 * r2) - 1 + 2 * (Math.log(REF_SIGMA) - policy.logSigma);
  }

  function sampleEnd(policy, normal) {
    const m = policyMean(policy);
    const s = Math.exp(policy.logSigma);
    return [m[0] + s * normal(), m[1] + s * normal()];
  }

  // the walk adds the offset a little each step, and per-step noise sigma sqrt(dt) gives endpoint spread sigma
  function sampleWalk(policy, normal) {
    const sd = Math.exp(policy.logSigma) * Math.sqrt(W.DT);
    const points = [START];
    for (let k = 0; k < W.STEPS; k++) {
      const p = points[k];
      points.push([
        p[0] + W.DRIFT[k][0] + policy.theta[0] * W.DT + sd * normal(),
        p[1] + W.DRIFT[k][1] + policy.theta[1] * W.DT + sd * normal(),
      ]);
    }
    return points;
  }

  // ---------- preferences ----------

  // how much a biased labeler also likes answers further right, a stand-in for longer answers
  const LENGTH_BIAS = 2.5;

  /** Return what the labeler compares: the true utility, plus a liking for length when biased. */
  // axis 0 is the length stand-in; an ai labeler's own taste uses axis 1
  function labelerScore(y, bias = 0, axis = 0) {
    return trueUtility(y) + bias * y[axis];
  }

  // the simulated human is noisy: it picks A with probability sigmoid((score(A) - score(B)) / noise)
  function simulatedLabel(a, b, uniform, bias = 0, noise = 1, axis = 0) {
    return uniform() < sigmoid((labelerScore(a, bias, axis) - labelerScore(b, bias, axis)) / noise) ? 0 : 1;
  }

  /** Return n labelled pairs {a, b, winner} sampled from policy (the reference by default). */
  function collectPairs(n, seed, { policy = refPolicy(), bias = 0, noise = 1, axis = 0 } = {}) {
    const uniform = seededUniform(seed);
    const normal = gaussianFrom(uniform);
    return Array.from({ length: n }, () => {
      const a = sampleEnd(policy, normal);
      const b = sampleEnd(policy, normal);
      return { a, b, winner: simulatedLabel(a, b, uniform, bias, noise, axis) };
    });
  }

  // ---------- reward model ----------

  // a deliberately simple reward model: a straight slope. it ranks pairs well where the data is,
  // but it keeps rising forever outside the data, which is what makes overoptimization show up
  const RIDGE = 0.01;

  function rmFeatures(y) {
    return [y[0] - REF_MEAN[0], y[1] - REF_MEAN[1]];
  }

  /** Fit a bradley-terry reward model to labelled pairs by newton's method with a ridge penalty. */
  function fitRewardModel(pairs) {
    const F = rmFeatures(REF_MEAN).length;
    const ridge = new Array(F).fill(RIDGE);
    let w = new Array(F).fill(0);
    if (!pairs.length) return { w };
    const diffs = pairs.map((p) => {
      const fa = rmFeatures(p.a);
      const fb = rmFeatures(p.b);
      const d = fa.map((x, i) => x - fb[i]);
      // orient each difference as winner minus loser
      return p.winner === 0 ? d : d.map((x) => -x);
    });
    for (let iter = 0; iter < 25; iter++) {
      const g = w.map((x, i) => ridge[i] * x);
      const H = Array.from({ length: F }, (_, i) => Array.from({ length: F }, (_, j) => (i === j ? ridge[i] : 0)));
      for (const d of diffs) {
        const z = d.reduce((s, x, i) => s + x * w[i], 0);
        const p = sigmoid(z);
        for (let i = 0; i < F; i++) {
          g[i] -= (1 - p) * d[i];
          for (let j = 0; j < F; j++) H[i][j] += p * (1 - p) * d[i] * d[j];
        }
      }
      const step = solve(H, g);
      w = w.map((x, i) => x - step[i]);
    }
    return { w };
  }

  function rmReward(model, y) {
    return rmFeatures(y).reduce((s, x, i) => s + x * model.w[i], 0);
  }

  /** Return the fraction of pairs where the reward model ranks the true utility's choice first. */
  function pairAccuracy(model, pairs) {
    if (!pairs.length) return NaN;
    let ok = 0;
    for (const p of pairs) {
      const modelPrefersA = rmReward(model, p.a) >= rmReward(model, p.b);
      const truePrefersA = trueUtility(p.a) >= trueUtility(p.b);
      if (modelPrefersA === truePrefersA) ok++;
    }
    return ok / pairs.length;
  }

  function solve(A, b) {
    const n = A.length;
    const M = A.map((row, i) => [...row, b[i]]);
    for (let i = 0; i < n; i++) {
      let piv = i;
      for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[piv][i])) piv = r;
      [M[i], M[piv]] = [M[piv], M[i]];
      for (let r = 0; r < n; r++) {
        if (r === i) continue;
        const f = M[r][i] / M[i][i];
        for (let j = i; j <= n; j++) M[r][j] -= f * M[i][j];
      }
    }
    return M.map((row, i) => row[n] / M[i][i]);
  }

  // ---------- optimizers ----------

  function adam(lr) {
    let m = [0, 0, 0];
    let v = [0, 0, 0];
    let t = 0;
    return (params, grad) => {
      t++;
      m = m.map((x, i) => 0.9 * x + 0.1 * grad[i]);
      v = v.map((x, i) => 0.999 * x + 0.001 * grad[i] ** 2);
      return params.map((p, i) => p + (lr * (m[i] / (1 - 0.9 ** t))) / (Math.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8));
    };
  }

  const pack = (policy) => [policy.theta[0], policy.theta[1], policy.logSigma];
  const unpack = (p) => ({ theta: [p[0], p[1]], logSigma: Math.max(Math.log(0.02), Math.min(Math.log(0.6), p[2])) });

  // ascent direction of log pi(y) in the packed parameters
  function scoreOf(policy, y) {
    const s2 = Math.exp(2 * policy.logSigma);
    const m = policyMean(policy);
    const d = [y[0] - m[0], y[1] - m[1]];
    return [d[0] / s2, d[1] / s2, (d[0] ** 2 + d[1] ** 2) / s2 - 2];
  }

  const BATCH = 128;

  /**
   * Return a REINFORCE trainer for reward(y) - beta * log(pi(y) / pi_ref(y)).
   * The returned step() advances one update and returns the new policy and its stats.
   */
  function reinforceTrainer({ reward, beta, seed = 1, lr = 0.01, init = refPolicy(), batch = BATCH }) {
    const normal = gaussianFrom(seededUniform(seed));
    const ref = refPolicy();
    const update = adam(lr);
    let policy = { theta: init.theta.slice(), logSigma: init.logSigma };
    return {
      get policy() { return policy; },
      step() {
        const ys = Array.from({ length: batch }, () => sampleEnd(policy, normal));
        const R = ys.map((y) => reward(y) - beta * (logProb(policy, y) - logProb(ref, y)));
        // a mean baseline cuts variance without biasing the gradient
        const base = R.reduce((s, x) => s + x, 0) / batch;
        const grad = [0, 0, 0];
        ys.forEach((y, i) => {
          const sc = scoreOf(policy, y);
          for (let j = 0; j < 3; j++) grad[j] += ((R[i] - base) * sc[j]) / batch;
        });
        policy = unpack(update(pack(policy), grad));
        return policy;
      },
    };
  }

  /** Return an IPO trainer: it fits each pair's log-ratio gap to 1/(2 tau) instead of pushing it without bound. */
  function ipoTrainer({ pairs, tau, lr = 0.02 }) {
    const ref = refPolicy();
    const update = adam(lr);
    let policy = refPolicy();
    const oriented = pairs.map((p) => (p.winner === 0 ? [p.a, p.b] : [p.b, p.a]));
    const target = 1 / (2 * tau);
    return {
      get policy() { return policy; },
      step() {
        const grad = [0, 0, 0];
        for (const [yw, yl] of oriented) {
          const h = logProb(policy, yw) - logProb(ref, yw) - logProb(policy, yl) + logProb(ref, yl);
          // descent on (h - target)^2 moves h toward the target from either side
          const k = (-2 * (h - target)) / oriented.length;
          const sw = scoreOf(policy, yw);
          const sl = scoreOf(policy, yl);
          for (let j = 0; j < 3; j++) grad[j] += k * (sw[j] - sl[j]);
        }
        policy = unpack(update(pack(policy), grad));
        return policy;
      },
    };
  }

  /** Return a DPO trainer on labelled pairs; its step() runs one full-batch update. */
  function dpoTrainer({ pairs, beta, lr = 0.02 }) {
    const ref = refPolicy();
    const update = adam(lr);
    let policy = refPolicy();
    const oriented = pairs.map((p) => (p.winner === 0 ? [p.a, p.b] : [p.b, p.a]));
    return {
      get policy() { return policy; },
      step() {
        const grad = [0, 0, 0];
        for (const [yw, yl] of oriented) {
          const h = beta * (logProb(policy, yw) - logProb(ref, yw) - logProb(policy, yl) + logProb(ref, yl));
          // ascent on log sigmoid(h) scales the score difference by 1 - sigmoid(h)
          const k = (beta * (1 - sigmoid(h))) / oriented.length;
          const sw = scoreOf(policy, yw);
          const sl = scoreOf(policy, yl);
          for (let j = 0; j < 3; j++) grad[j] += k * (sw[j] - sl[j]);
        }
        policy = unpack(update(pack(policy), grad));
        return policy;
      },
    };
  }

  /** Return mean reward, mean true utility, and verifier pass rate of a policy, by a fixed-seed estimate. */
  function evaluate(policy, model, n = 2000) {
    const normal = gaussianFrom(seededUniform(99));
    let proxy = 0;
    let utility = 0;
    let pass = 0;
    for (let i = 0; i < n; i++) {
      const y = sampleEnd(policy, normal);
      if (model) proxy += rmReward(model, y);
      utility += trueUtility(y);
      pass += verifierReward(y);
    }
    return { proxy: proxy / n, utility: utility / n, pass: pass / n, kl: klToRef(policy), sigma: Math.exp(policy.logSigma) };
  }

  function train(trainer, steps) {
    for (let i = 0; i < steps; i++) trainer.step();
    return trainer.policy;
  }

  // ---------- defenses against reward hacking ----------

  /** Fit k reward models, each on a bootstrap resample of the pairs. */
  function fitEnsemble(pairs, k, seed) {
    const uniform = seededUniform(seed);
    return Array.from({ length: k }, () =>
      fitRewardModel(pairs.map(() => pairs[Math.floor(uniform() * pairs.length)])),
    );
  }

  // members agree where the data is and disagree away from it, so mean minus spread is cautious there
  function conservativeReward(models, caution) {
    return (y) => {
      const r = models.map((m) => rmReward(m, y));
      const mean = r.reduce((s, x) => s + x, 0) / r.length;
      const sd = Math.sqrt(r.reduce((s, x) => s + (x - mean) ** 2, 0) / r.length);
      return mean - caution * sd;
    };
  }

  /**
   * Run one round of iterated RLHF: label fresh pairs from the current policy, refit the reward model
   * on that round only, and continue RL from the current policy. Return {policy, model, pairs}.
   */
  function iterateRound(policy, { beta, bias = 0, pairs = 150, steps = 150, seed = 1 }) {
    const fresh = collectPairs(pairs, seed, { policy, bias });
    const model = fitRewardModel(fresh);
    const trainer = reinforceTrainer({ reward: (y) => rmReward(model, y), beta, seed, init: policy });
    return { policy: train(trainer, steps), model, pairs: fresh };
  }

  const BETAS = [3, 1.5, 0.8, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05];

  /** Return the end-of-training stats of RLHF for every beta in BETAS. */
  function betaSweep(model, steps = 300) {
    return BETAS.map((beta) => {
      const policy = train(reinforceTrainer({ reward: (y) => rmReward(model, y), beta, seed: 5 }), steps);
      return { beta, ...evaluate(policy, model) };
    });
  }

  const api = {
    START, VIEW, REF_SIGMA, REF_MEAN, UTILITY, VERIFIER, BETAS, LENGTH_BIAS, labelerScore,
    fitEnsemble, conservativeReward, iterateRound,
    seededUniform, sigmoid, trueUtility, verifierReward,
    refPolicy, policyMean, logProb, klToRef, sampleEnd, sampleWalk,
    simulatedLabel, collectPairs, fitRewardModel, rmReward, pairAccuracy,
    reinforceTrainer, dpoTrainer, ipoTrainer, evaluate, train, betaSweep,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.RLHF = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
