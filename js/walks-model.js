// pure math for the reasoning-as-pathfinding figures; no DOM, so tools/check-model.js can run it in node.
// a walk stands in for one chain of thought: q[k+1] = q[k] + drift[k] + offset * dt + noise.
// the drift is the base model's habit; the offset is the only thing training changes.
(function (root) {
  "use strict";

  const STEPS = 40;
  const DT = 1 / STEPS;
  const SPEED = 1.35;
  const TURN = 1.2;
  const HEADING = 0.6;
  const ROUNDS = 8;
  const BATCH = 64;
  // small ridge keeps the normal equations solvable when few endpoints land
  const RIDGE = 1e-3;

  // a chain longer than STEPS keeps turning at the same rate, so extra steps carry it past the target
  function driftAt(k) {
    const angle = HEADING - TURN * (k + 0.5) * DT;
    return [SPEED * Math.cos(angle) * DT, SPEED * Math.sin(angle) * DT];
  }
  const DRIFT = Array.from({ length: STEPS }, (_, k) => driftAt(k));
  const arcs = new Map();
  /** Return where the drift alone carries a walk of the given number of steps. */
  function arcFor(steps) {
    if (!arcs.has(steps)) {
      let a = [0, 0];
      for (let k = 0; k < steps; k++) {
        const d = driftAt(k);
        a = [a[0] + d[0], a[1] + d[1]];
      }
      arcs.set(steps, a);
    }
    return arcs.get(steps);
  }
  // optional variant fields: steps (chain length), noise (multiplies diffusion), batch (samples per round)
  const stepsOf = (variant) => variant.steps || STEPS;
  const diffusionOf = (variant) => variant.diffusion * (variant.noise || 1);

  // seeded so training gives the same weights on every page load
  function seededUniform(seed) {
    let state = seed | 0;
    return () => {
      state = (state + 0x6d2b79f5) | 0;
      let t = Math.imul(state ^ (state >>> 15), 1 | state);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gaussianFrom(uniform) {
    return () => Math.sqrt(-2 * Math.log(uniform() + 1e-12)) * Math.cos(2 * Math.PI * uniform());
  }

  const VARIANTS = {
    walks: {
      diffusion: 0.008,
      // tall enough for the longest chain-length setting, which curls down past the target
      view: [-1.12, 1.3, -1.45, 0.5],
      prompts: [{ start: [-0.75, 0], regions: [{ kind: "correct", center: [0.5, -0.26], r: 0.2 }] }],
    },
    amortize: {
      diffusion: 0.003,
      view: [-1.2, 1.6, -1.05, 1.3],
      train: true,
      draggable: true,
      bumpWidth: 0.4,
      prompts: [-0.9, -0.3, 0.3, 0.9]
        .map((y) => ({ start: [-0.9, y] }))
        .concat([{ start: [-0.6, 0.6], isNew: true }]),
      // every prompt gets its own target, placed off the base arc so the untrained model misses it
      regionsFor: (start) => [{ kind: "correct", center: [start[0] + 1.37, 0.6 * start[1] - 0.1], r: 0.13 }],
    },
    verifier: {
      diffusion: 0.008,
      view: [-1, 1, -1, 0.5],
      train: true,
      imperfectToggle: true,
      prompts: [
        {
          start: [-0.75, 0],
          regions: [
            { kind: "correct", center: [0.55, -0.4], r: 0.2 },
            { kind: "accepted", center: [0.263, -0.312], semiAxes: [0.4, 0.2], angle: 2.84 },
          ],
        },
      ],
    },
  };

  function regionsOf(variant, prompt) {
    return prompt.regions || variant.regionsFor(prompt.start);
  }

  /** Return the distance from a region's center to its edge in direction angle. */
  function radiusAt(region, angle) {
    if (!region.semiAxes) return region.r;
    const [long, short] = region.semiAxes;
    const u = Math.cos(angle - region.angle) / long;
    const w = Math.sin(angle - region.angle) / short;
    return 1 / Math.sqrt(u * u + w * w);
  }

  function contains(region, q) {
    const dx = q[0] - region.center[0];
    const dy = q[1] - region.center[1];
    return Math.hypot(dx, dy) <= radiusAt(region, Math.atan2(dy, dx));
  }

  function isCorrect(regions, q) {
    return regions.some((g) => g.kind === "correct" && contains(g, q));
  }

  // an imperfect verifier replaces the correct set with its own accepted set
  function verifierAccepts(regions, q, imperfect) {
    const accepted = imperfect && regions.find((g) => g.kind === "accepted");
    return accepted ? contains(accepted, q) : isCorrect(regions, q);
  }

  // the filter for which samples training keeps: by default the verifier, or with keepScale a disk scaled
  // around each correct region's center, so a stricter or looser threshold than "lands" (rest-style)
  function keeps(variant, regions, q, imperfect) {
    if (!variant.keepScale) return verifierAccepts(regions, q, imperfect);
    return regions.some((g) => g.kind === "correct" && Math.hypot(q[0] - g.center[0], q[1] - g.center[1]) <= g.r * variant.keepScale);
  }

  // amortize uses one gaussian bump per training prompt, so what is learned fades with distance from them
  function features(variant, start) {
    if (!variant.bumpWidth) return [start[1], 1];
    const w2 = 2 * variant.bumpWidth ** 2;
    return variant.prompts
      .filter((p) => !p.isNew)
      .map((p) => Math.exp(-((start[0] - p.start[0]) ** 2 + (start[1] - p.start[1]) ** 2) / w2));
  }

  function zeroWeights(variant) {
    return features(variant, [0, 0]).map(() => [0, 0]);
  }

  function offset(variant, weights, start) {
    return features(variant, start).reduce(
      (o, f, j) => [o[0] + f * weights[j][0], o[1] + f * weights[j][1]],
      [0, 0],
    );
  }

  function endpointMean(variant, weights, start) {
    const o = offset(variant, weights, start);
    const steps = stepsOf(variant);
    const arc = arcFor(steps);
    const t = DT * steps;
    return [start[0] + arc[0] + o[0] * t, start[1] + arc[1] + o[1] * t];
  }

  function endpointSd(variant) {
    return Math.sqrt(2 * diffusionOf(variant) * DT * stepsOf(variant));
  }

  /** Solve A X = B by Gauss-Jordan elimination with partial pivoting. */
  function solve(A, B) {
    const n = A.length;
    const M = A.map((row, i) => [...row, ...B[i]]);
    for (let i = 0; i < n; i++) {
      let pivot = i;
      for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[pivot][i])) pivot = r;
      [M[i], M[pivot]] = [M[pivot], M[i]];
      for (let r = 0; r < n; r++) {
        if (r === i) continue;
        const f = M[r][i] / M[i][i];
        for (let j = i; j < M[r].length; j++) M[r][j] -= f * M[i][j];
      }
    }
    return M.map((row, i) => row.slice(n).map((x) => x / M[i][i]));
  }

  // least squares on kept endpoints is maximum likelihood for gaussian walks, so this is one rsft step
  function fitRound(variant, weights, normal, imperfect) {
    const F = weights.length;
    const sd = endpointSd(variant);
    const A = Array.from({ length: F }, (_, i) => Array.from({ length: F }, (_, j) => (i === j ? RIDGE : 0)));
    const B = Array.from({ length: F }, () => [0, 0]);
    let kept = 0;
    for (const prompt of variant.prompts.filter((p) => !p.isNew)) {
      const mean = endpointMean(variant, weights, prompt.start);
      const f = features(variant, prompt.start);
      const regions = regionsOf(variant, prompt);
      for (let n = 0; n < (variant.batch || BATCH); n++) {
        const q = [mean[0] + sd * normal(), mean[1] + sd * normal()];
        if (!keeps(variant, regions, q, imperfect)) continue;
        kept++;
        const arc = arcFor(stepsOf(variant));
        const t = DT * stepsOf(variant);
        const excess = [(q[0] - prompt.start[0] - arc[0]) / t, (q[1] - prompt.start[1] - arc[1]) / t];
        for (let i = 0; i < F; i++) {
          for (let j = 0; j < F; j++) A[i][j] += f[i] * f[j];
          B[i][0] += f[i] * excess[0];
          B[i][1] += f[i] * excess[1];
        }
      }
    }
    return kept ? solve(A, B) : weights;
  }

  /** Return the weights after each training round, round 0 being the untrained model. */
  function train(variant, imperfect) {
    const normal = gaussianFrom(seededUniform(imperfect ? 7 : 11));
    const history = [zeroWeights(variant)];
    for (let t = 0; t < ROUNDS; t++) history.push(fitRound(variant, history[t], normal, imperfect));
    return history;
  }

  // a fixed noise cloud, so the displayed percentages do not flicker between frames
  const CLOUD = (() => {
    const normal = gaussianFrom(seededUniform(3));
    return Array.from({ length: 4000 }, () => [normal(), normal()]);
  })();

  /** Return the fraction of endpoints with the given outcome: accepted, correct, or wrong (accepted but not correct). */
  function successRate(variant, prompt, weights, imperfect, outcome = "accepted") {
    const mean = endpointMean(variant, weights, prompt.start);
    const sd = endpointSd(variant);
    const regions = regionsOf(variant, prompt);
    let count = 0;
    for (const [gx, gy] of CLOUD) {
      const q = [mean[0] + sd * gx, mean[1] + sd * gy];
      const correct = isCorrect(regions, q);
      const accepted = verifierAccepts(regions, q, imperfect);
      if (outcome === "correct" ? correct : outcome === "wrong" ? accepted && !correct : accepted) count++;
    }
    return count / CLOUD.length;
  }

  // e^-x I0(x), from Abramowitz and Stegun 9.8.1 and 9.8.2
  function scaledI0(x) {
    if (x <= 3.75) {
      const t = (x / 3.75) ** 2;
      return Math.exp(-x) * (1 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813))))));
    }
    const t = 3.75 / x;
    return (0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565 + t * (0.00916281 +
      t * (-0.02057706 + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377)))))))) / Math.sqrt(x);
  }

  /** Return the exact chance that a round gaussian with this sd, centred d away from a disk of radius r, lands in it. */
  function diskProbability(d, r, sd) {
    const s2 = sd * sd;
    const n = 40;
    const f = (rho) => (rho / s2) * Math.exp(-((rho - d) ** 2) / (2 * s2)) * scaledI0((rho * d) / s2);
    let sum = f(0) + f(r);
    for (let i = 1; i < n; i++) sum += (i % 2 ? 4 : 2) * f((r * i) / n);
    return Math.min(1, (sum * r) / (3 * n));
  }

  /**
   * Return a ny-by-nx grid (row 0 at y0) of one sample's landing rate for a prompt placed at each point,
   * after training; the prompt's target comes from regionsFor, so this works only for the amortize variant.
   */
  function landingField(variant, weights, [x0, x1, y0, y1], nx, ny) {
    const out = new Float32Array(nx * ny);
    const sd = endpointSd(variant);
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const start = [x0 + ((x1 - x0) * i) / (nx - 1), y0 + ((y1 - y0) * j) / (ny - 1)];
        const mean = endpointMean(variant, weights, start);
        const g = variant.regionsFor(start)[0];
        out[j * nx + i] = diskProbability(Math.hypot(mean[0] - g.center[0], mean[1] - g.center[1]), g.r, sd);
      }
    }
    return out;
  }

  // every correct answer is the same answer (like "55"); wrong answers differ by which grid cell they fall in,
  // the way different mistakes give different numbers
  const CELL = 0.15;

  /**
   * Return the success rate of majority voting over k samples for each k in ks: the most common answer
   * wins, and the vote succeeds if it is the correct answer. No verifier is used.
   */
  function majorityRates(variant, prompt, weights, ks, trials = 1500) {
    const mean = endpointMean(variant, weights, prompt.start);
    const sd = endpointSd(variant);
    const regions = regionsOf(variant, prompt);
    const counts = new Map();
    for (const [gx, gy] of CLOUD) {
      const q = [mean[0] + sd * gx, mean[1] + sd * gy];
      const key = isCorrect(regions, q) ? "correct" : `${Math.floor(q[0] / CELL)},${Math.floor(q[1] / CELL)}`;
      counts.set(key, (counts.get(key) || 0) + 1);
    }
    const cells = [...counts.keys()];
    const cum = [];
    let acc = 0;
    for (const c of cells) {
      acc += counts.get(c) / CLOUD.length;
      cum.push(acc);
    }
    const good = cells.map((c) => c === "correct");
    const uniform = seededUniform(17);
    const draw = () => {
      const u = uniform();
      let i = 0;
      while (i < cum.length - 1 && cum[i] < u) i++;
      return i;
    };
    return ks.map((k) => {
      let wins = 0;
      for (let t = 0; t < trials; t++) {
        const tally = new Map();
        for (let j = 0; j < k; j++) {
          const i = draw();
          tally.set(i, (tally.get(i) || 0) + 1);
        }
        let best = -1;
        let top = [];
        for (const [i, n] of tally) {
          if (n > best) [best, top] = [n, [i]];
          else if (n === best) top.push(i);
        }
        // a tie is broken at random, as a coin flip between equally common answers
        if (good[top[Math.floor(uniform() * top.length)]]) wins++;
      }
      return wins / trials;
    });
  }

  // per-step noise variance is 2 D dt, so the endpoint variance matches endpointSd
  function sampleWalk(variant, prompt, weights, normal) {
    const o = offset(variant, weights, prompt.start);
    const sd = Math.sqrt(2 * diffusionOf(variant) * DT);
    const points = [prompt.start];
    for (let k = 0; k < stepsOf(variant); k++) {
      const prev = points[k];
      const d = driftAt(k);
      points.push([
        prev[0] + d[0] + o[0] * DT + sd * normal(),
        prev[1] + d[1] + o[1] * DT + sd * normal(),
      ]);
    }
    return points;
  }

  const api = {
    STEPS, DT, ROUNDS, HEADING, TURN, DRIFT, VARIANTS, arcFor, stepsOf, driftAt,
    regionsOf, radiusAt, contains, isCorrect, verifierAccepts,
    zeroWeights, train, successRate, sampleWalk, gaussianFrom, majorityRates, landingField, offset,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ReasoningWalks = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
