// run with: node tools/check-rlhf.js
const assert = require("node:assert");
const R = require("../js/rlhf-model.js");

const f = (x) => x.toFixed(2);
const pairs = R.collectPairs(200, 1);
const held = R.collectPairs(500, 2);
const model = R.fitRewardModel(pairs);
const acc = R.pairAccuracy(model, held);
console.log(`reward model: held-out pair accuracy ${f(acc)}`);
assert(acc > 0.8, "reward model should rank most held-out pairs like the hidden utility");

const ref = R.evaluate(R.refPolicy(), model);
const sweep = R.betaSweep(model);
const best = sweep.reduce((a, b) => (b.utility > a.utility ? b : a));
const smallest = sweep[sweep.length - 1];
console.log(`reference: utility ${f(ref.utility)}`);
for (const s of sweep) console.log(`beta ${s.beta}: proxy ${f(s.proxy)} utility ${f(s.utility)} kl ${f(s.kl)} sigma ${s.sigma.toFixed(3)}`);
assert(best.utility > ref.utility + 0.5, "a moderate beta should raise the true utility");
assert(smallest.utility < best.utility - 1, "a tiny beta should overoptimize the proxy and lose true utility");
assert(smallest.proxy > best.proxy, "the proxy should keep rising as beta falls");
assert(smallest.kl > best.kl, "kl should grow as beta falls");

const dpo = R.evaluate(R.train(R.dpoTrainer({ pairs, beta: 0.3 }), 300), model);
console.log(`dpo beta 0.3: utility ${f(dpo.utility)} kl ${f(dpo.kl)}`);
assert(dpo.utility > ref.utility + 0.5, "dpo should raise the true utility");

const rlvr = R.evaluate(R.train(R.reinforceTrainer({ reward: (y) => R.UTILITY.height * R.verifierReward(y), beta: 0.3, seed: 5 }), 300), null);
console.log(`rlvr beta 0.3: pass ${f(rlvr.pass)} utility ${f(rlvr.utility)} sigma ${rlvr.sigma.toFixed(3)}`);
assert(rlvr.pass > 0.8, "rlvr should learn to land in the verifier region");

console.log("ok");

// step 8 uses these exact seeds, so these checks cover what the page claims
const HACK_SEED = 7;
const refU = R.evaluate(R.refPolicy(), null, 1000).utility;
const outcome = (policy) => ({
  gain: R.evaluate(policy, null, 1000).utility - refU,
  dx: R.policyMean(policy)[0] - R.UTILITY.center[0],
});
const plainRun = (pairs, beta) => {
  const m = R.fitRewardModel(pairs);
  return outcome(R.train(R.reinforceTrainer({ reward: (y) => R.rmReward(m, y), beta, seed: 5 }), 300));
};
const fair = plainRun(R.collectPairs(300, HACK_SEED, { bias: 0 }), 0.3);
const biased = plainRun(R.collectPairs(300, HACK_SEED, { bias: R.LENGTH_BIAS }), 0.3);
console.log(`spurious: fair gain ${f(fair.gain)} dx ${f(fair.dx)}; biased gain ${f(biased.gain)} dx ${f(biased.dx)}`);
assert(biased.dx > fair.dx + 0.15, "biased labels should push the policy toward longer answers");
assert(biased.gain < fair.gain - 0.3, "biased labels should cost true utility");

function defenses(bias) {
  const pairs = R.collectPairs(300, HACK_SEED, { bias });
  const ens = R.conservativeReward(R.fitEnsemble(pairs, 8, HACK_SEED), 3);
  let iter = R.refPolicy();
  for (let r = 1; r <= 10; r++) iter = R.iterateRound(iter, { beta: 0.1, bias, steps: 15, seed: HACK_SEED * 100 + r }).policy;
  return {
    plain: plainRun(pairs, 0.1),
    ensemble: outcome(R.train(R.reinforceTrainer({ reward: ens, beta: 0.1, seed: 5 }), 300)),
    iterated: outcome(iter),
  };
}
const d0 = defenses(0);
const d2 = defenses(2);
const d5 = defenses(5);
for (const [name, d] of [["bias 0", d0], ["bias 2", d2], ["bias 5", d5]]) {
  console.log(`${name}: plain ${f(d.plain.gain)} ensemble ${f(d.ensemble.gain)} iterated ${f(d.iterated.gain)}`);
}
assert(d0.ensemble.gain > d0.plain.gain + 0.5, "with fair labels the ensemble should stop the runaway");
assert(d0.iterated.gain > d0.plain.gain + 0.5, "with fair labels iterated rlhf should stop the runaway");
assert(d2.ensemble.gain < 0.5, "a shared bias should defeat the ensemble");
assert(d2.iterated.gain > 0.8, "iterated rlhf should still work at a small bias");
assert(d5.iterated.gain < 0.5, "a large bias should defeat iterated rlhf too");
console.log("step 8 ok");

// step 6 quotes these numbers about how much dpo depends on the label set
{
  const rows = [];
  for (let seed = 10; seed < 30; seed++) {
    const pairs = R.collectPairs(200, seed);
    const m = R.fitRewardModel(pairs);
    const rl = R.evaluate(R.train(R.reinforceTrainer({ reward: (y) => R.rmReward(m, y), beta: 0.3, seed: 5 }), 300), m);
    const d = R.evaluate(R.train(R.dpoTrainer({ pairs, beta: 0.3 }), 300), m);
    rows.push({ rl: rl.utility - ref.utility, dpo: d.utility - ref.utility, spread: d.sigma / R.REF_SIGMA });
  }
  const avg = (k) => rows.reduce((s, r) => s + r[k], 0) / rows.length;
  const spreads = rows.map((r) => r.spread);
  console.log(`dpo over 20 label sets: spread ${f(Math.min(...spreads))}-${f(Math.max(...spreads))}, gain ${f(avg("dpo"))} vs rlhf ${f(avg("rl"))}`);
  assert(Math.abs(Math.min(...spreads) - 0.75) < 0.03 && Math.abs(Math.max(...spreads) - 1.55) < 0.03, "step 6 quotes a 75% to 155% spread range");
  assert(Math.abs(avg("dpo") - 0.82) < 0.02 && Math.abs(avg("rl") - 1.0) < 0.02, "step 6 quotes gains of 0.82 and 1.00");
  console.log("step 6 ok");
}

// step 5 says three or four rounds of labelling from the policy bring a runaway policy back
{
  let pairs = R.collectPairs(200, 1);
  let model = R.fitRewardModel(pairs);
  const t = R.reinforceTrainer({ reward: (y) => R.rmReward(model, y), beta: 0.05, seed: 5 });
  for (let i = 0; i < 300; i++) t.step();
  const before = R.evaluate(t.policy, null, 800);
  for (let r = 1; r <= 4; r++) {
    pairs = pairs.concat(R.collectPairs(100, 100 + r, { policy: t.policy }));
    model = R.fitRewardModel(pairs);
    for (let i = 0; i < 300; i++) t.step();
  }
  const after = R.evaluate(t.policy, null, 800);
  console.log(`labels from the policy: utility ${f(before.utility)} -> ${f(after.utility)}, kl ${f(before.kl)} -> ${f(after.kl)}`);
  assert(before.utility < 0.1 && after.utility > 1.2 && after.kl < before.kl / 5, "four rounds of policy labels should pull the policy back");
  console.log("step 5 ok");
}

// step 7 says a verifier radius of 0.10 or less gives rlvr almost no passing samples to learn from
{
  const passAt = (r) => {
    R.VERIFIER.r = r;
    return R.evaluate(R.train(R.reinforceTrainer({ reward: (y) => R.UTILITY.height * R.verifierReward(y), beta: 0.3, seed: 5 }), 300), null).pass;
  };
  const small = [0.05, 0.1].map(passAt);
  const normal16 = passAt(0.16);
  R.VERIFIER.r = 0.16;
  console.log(`rlvr pass rate at radius 0.05 / 0.10 / 0.16: ${small.map(f).join(" / ")} / ${f(normal16)}`);
  assert(small.every((p) => p <= 0.03) && normal16 > 0.9, "a small verifier should starve rlvr of signal");
  console.log("step 7 ok");
}

// step 4 ai labeler and step 6 ipo against dpo
{
  const held2 = R.collectPairs(500, 2);
  const ref2 = R.evaluate(R.refPolicy(), null, 2000).utility;
  const viaRL = (pairs) => {
    const m = R.fitRewardModel(pairs);
    return { acc: R.pairAccuracy(m, held2), gain: R.evaluate(R.train(R.reinforceTrainer({ reward: (y) => R.rmReward(m, y), beta: 0.3, seed: 5 }), 300), null, 2000).utility - ref2 };
  };
  const ai = viaRL(R.collectPairs(200, 1, { bias: 1.5, noise: 0.5, axis: 1 }));
  const ai10 = viaRL(R.collectPairs(2000, 1, { bias: 1.5, noise: 0.5, axis: 1 }));
  console.log(`ai labeler: accuracy ${f(ai.acc)}, rl gain ${f(ai.gain)}; with 2000 labels ${f(ai10.gain)}`);
  assert(ai.acc > 0.9 && ai.gain < 0 && ai10.gain < 0, "ai labels should rank pairs well yet send rl the wrong way");

  const klOf = (trainer, steps) => R.evaluate(R.train(trainer, steps), null, 2000).kl;
  const dpo1 = klOf(R.dpoTrainer({ pairs: R.collectPairs(200, 1, { noise: 1 }), beta: 0.3 }), 600);
  const dpo01 = klOf(R.dpoTrainer({ pairs: R.collectPairs(200, 1, { noise: 0.1 }), beta: 0.3 }), 600);
  const ipo1 = klOf(R.ipoTrainer({ pairs: R.collectPairs(200, 1, { noise: 1 }), tau: 0.3 }), 600);
  const ipo01 = klOf(R.ipoTrainer({ pairs: R.collectPairs(200, 1, { noise: 0.1 }), tau: 0.3 }), 600);
  console.log(`kl, labeler noise 1 / 0.1: dpo ${f(dpo1)} / ${f(dpo01)}, ipo ${f(ipo1)} / ${f(ipo01)}`);
  assert(Math.abs(dpo1 - 1.8) < 0.1 && Math.abs(dpo01 - 3.1) < 0.1, "step 6 quotes dpo kl 1.8 to 3.1");
  assert(ipo1 < 0.15 && ipo01 < 0.25, "step 6 quotes ipo kl 0.1 to 0.2");
  console.log("step 4 and 6 good fits ok");
}
