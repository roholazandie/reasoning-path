// run with: node tools/check-beyond.js
const assert = require("node:assert");
const B = require("../js/beyond-model.js");
const R = require("../js/rlhf-model.js");

const pct = (x) => `${(100 * x).toFixed(0)}%`;
const f = (x) => (Number.isFinite(x) ? x.toFixed(1) : "-");
const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;

// step 11: rl raises pass@1 and lowers pass@k at large k
const ks = [1, 16, 256];
const base = { theta: [0, 0], logSigma: Math.log(B.COVERAGE.baseSigma) };
const trained = B.trainCoverage({}).at(-1);
const pb = B.passAtK(base, ks);
const pt = B.passAtK(trained, ks);
console.log(`pass@1/16/256 base ${pb.map(pct).join(" ")}, trained ${pt.map(pct).join(" ")}`);
assert(pt[0] > 3 * pb[0], "rl should raise pass@1 a lot");
assert(pb[1] > pt[1] && pb[2] > pt[2] + 0.4, "the base model should win at large k");
const wide = B.passAtK(B.trainCoverage({ beta: 0.3 }).at(-1), ks);
console.log(`with a narrowing penalty of 0.3: ${wide.map(pct).join(" ")}`);
assert(wide[2] > pt[2] + 0.4 && wide[0] < pt[0], "a narrowing penalty should trade pass@1 for coverage");

// step 11: process reward learns faster at first, then stalls below the outcome reward
{
  const curve = (reward) => {
    const runs = [1, 2, 3, 4].map((seed) => B.trainProcess({ reward, seed }));
    return runs[0].map((_, i) => mean(runs.map((h) => h[i].pass)));
  };
  const out = curve("outcome");
  const proc = curve("process");
  console.log(`landing at step 60 / 300: outcome ${pct(out[6])} / ${pct(out[30])}, process ${pct(proc[6])} / ${pct(proc[30])}`);
  assert(proc[6] > 2 * out[6], "process reward should learn faster early");
  assert(out[30] > 0.9 && proc[30] < 0.7, "outcome should end near 95% and process stall near 60%");
}

// step 12: the grpo length bias, averaged like the page (4 runs, last 50 steps)
const tail = (runs, key) => {
  const per = runs[0].history.map((_, i) => mean(runs.map((r) => r.history[i][key]).filter(Number.isFinite)));
  return mean(per.slice(-50).filter(Number.isFinite));
};
const runsFor = (algorithm, lr = 0.02) => [1, 2, 3, 4].map((seed) => B.trainLength({ algorithm, seed, lr }));
const gap = (runs) => tail(runs, "lenWrong") - tail(runs, "lenCorrect");
const grpo = runsFor("grpo");
const dr = runsFor("drgrpo");
const rf = runsFor("reinforce");
const rloo = runsFor("rloo");
console.log(`wrong minus right length: grpo ${f(gap(grpo))}, dr. grpo ${f(gap(dr))}, reinforce ${f(gap(rf))}, rloo ${f(gap(rloo))}`);
assert(gap(grpo) > 3 && gap(dr) < -3, "grpo should leave wrong answers longer, dr. grpo shorter");
assert(Math.abs(gap(rf) - gap(rloo)) < 1e-9, "rloo and reinforce with a mean baseline should give the same run");
// same update up to a constant; rounding makes the sampled answers drift apart later, so only roughly equal
assert(Math.abs(gap(rf) - gap(dr)) < 1.5 && Math.sign(gap(rf)) === Math.sign(gap(dr)), "dr. grpo should behave like reinforce");
const hi = Object.fromEntries(["reinforce", "grpo", "ppo"].map((a) => [a, tail(runsFor(a, 0.1), "success")]));
console.log(`success at learning rate 0.1: reinforce ${pct(hi.reinforce)}, grpo ${pct(hi.grpo)}, ppo ${pct(hi.ppo)}`);
assert(hi.ppo > 2 * hi.reinforce && hi.ppo > 2 * hi.grpo, "ppo-clip should hold up best at a high learning rate");

// step 12: best-of-n against rl at the same kl
const model = R.fitRewardModel(R.collectPairs(200, 1));
const ref0 = R.evaluate(R.refPolicy(), model, 4000).utility;
const rlPeak = Math.max(...R.betaSweep(model).map((s) => s.utility - ref0));
const uniform = R.seededUniform(5);
const g = () => Math.sqrt(-2 * Math.log(uniform() + 1e-12)) * Math.cos(2 * Math.PI * uniform());
const bonGain = (N, trials) => {
  let u = 0;
  for (let t = 0; t < trials; t++) {
    let best = null;
    let br = -Infinity;
    for (let i = 0; i < N; i++) {
      const y = R.sampleEnd(R.refPolicy(), g);
      const r = R.rmReward(model, y);
      if (r > br) [br, best] = [r, y];
    }
    u += R.trueUtility(best);
  }
  return u / trials - ref0;
};
const b64 = bonGain(64, 1200);
const b4096 = bonGain(4096, 250);
console.log(`true gain: rl peak ${f(rlPeak)}, best-of-64 ${f(b64)}, best-of-4096 ${f(b4096)}`);
assert(b64 > rlPeak && b4096 < b64 - 0.3, "best-of-n should beat rl's peak and then overoptimize");
console.log("ok");

// step 12: chains grow during training
{
  const lenAt = (algorithm) => {
    const runs = [1, 2, 3, 4].map((seed) => B.trainLength({ algorithm, seed }).history);
    return runs.reduce((s, h) => s + h[299].muL, 0) / runs.length;
  };
  const lr0 = lenAt("reinforce");
  const lp = lenAt("ppo");
  console.log(`mean chain length after training: reinforce ${f(lr0)}, ppo ${f(lp)}`);
  assert(Math.abs(lr0 - 44) < 1 && Math.abs(lp - 47) < 1, "step 12 quotes about 44 and 47 steps");
  console.log("length growth ok");
}
