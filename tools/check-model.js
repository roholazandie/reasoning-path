// run with: node tools/check-model.js
const assert = require("node:assert");
const M = require("../js/walks-model.js");

const pct = (x) => `${(100 * x).toFixed(1)}%`;

const walks = M.VARIANTS.walks;
const pWalks = M.successRate(walks, walks.prompts[0], M.zeroWeights(walks), false);
console.log(`walks: untrained success ${pct(pWalks)}`);
assert(pWalks > 0.05 && pWalks < 0.95, "walks figure should show both hits and misses");

const am = M.VARIANTS.amortize;
const amRuns = M.train(am, true);
for (const p of am.prompts) {
  const before = M.successRate(am, p, amRuns[0], true);
  const after = M.successRate(am, p, amRuns[M.ROUNDS], true);
  console.log(`amortize ${p.isNew ? "new  " : "train"} ${p.start}: ${pct(before)} -> ${pct(after)}`);
  assert(after > before, "training should raise success for every prompt near the training set");
}
const far = { start: [-1.1, -1.0], isNew: true };
const nearAfter = M.successRate(am, am.prompts[4], amRuns[M.ROUNDS], true);
const farAfter = M.successRate(am, far, amRuns[M.ROUNDS], true);
console.log(`amortize far new prompt after training: ${pct(farAfter)}`);
assert(farAfter < nearAfter, "a far new prompt should generalize worse");

const ver = M.VARIANTS.verifier;
const p = ver.prompts[0];
const imperfect = M.train(ver, true)[M.ROUNDS];
const exact = M.train(ver, false)[M.ROUNDS];
const wrong = M.successRate(ver, p, imperfect, true, "wrong");
const correctImp = M.successRate(ver, p, imperfect, true, "correct");
const correctExact = M.successRate(ver, p, exact, false, "correct");
console.log(`verifier imperfect: wrong ${pct(wrong)}, correct ${pct(correctImp)}; exact: correct ${pct(correctExact)}`);
assert(wrong > correctImp, "imperfect verifier should pull walks onto wrong answers");
assert(correctExact > 0.5, "exact verifier should pull most walks onto correct answers");

console.log("ok");

// claims made by the step 1 to 3 sliders
{
  const base = M.VARIANTS.walks;
  const one = (v) => M.successRate(v, v.prompts[0], M.zeroWeights(v), true);
  const byLength = [30, 40, 45, 50, 60].map((n) => one({ ...base, steps: n }));
  console.log(`chain length 30/40/45/50/60: ${byLength.map(pct).join(" ")}`);
  assert(byLength[2] > byLength[1] && byLength[2] > byLength[0] && byLength[4] < byLength[1], "success should rise then fall with chain length");

  const am2 = M.VARIANTS.amortize;
  const newAfter = (bw) => {
    const v = { ...am2, bumpWidth: bw };
    return M.successRate(v, v.prompts[4], M.train(v, true)[M.ROUNDS], true);
  };
  console.log(`new prompt after training, width 0.1 / 1.2: ${pct(newAfter(0.1))} / ${pct(newAfter(1.2))}`);
  assert(newAfter(0.1) < 0.05 && newAfter(1.2) > 0.6, "generalization width should decide whether the new prompt improves");
  {
    const v = { ...am2, bumpWidth: 0.1 };
    const top = M.successRate(v, v.prompts[3], M.train(v, true)[M.ROUNDS], true);
    console.log(`top training prompt after training at width 0.1: ${pct(top)}`);
    assert(top < 0.05, "step 2 says the top prompt stays at 0% when the width is narrow");
  }

  const trainAvg = (batch) => {
    const v = { ...am2, batch };
    const w = M.train(v, true)[M.ROUNDS];
    return v.prompts.filter((p) => !p.isNew).reduce((s, p) => s + M.successRate(v, p, w, true), 0) / 4;
  };
  console.log(`training prompts after 8 rounds, 4 / 64 samples: ${pct(trainAvg(4))} / ${pct(trainAvg(64))}`);
  assert(trainAvg(4) < 0.35 && trainAvg(64) > 0.65, "very few samples per round should stall training");

  const ellipse = (length) => {
    const v = { ...ver, prompts: ver.prompts.map((q) => ({ ...q, regions: q.regions.map((g) => ({ ...g, semiAxes: g.semiAxes && g.semiAxes.slice() })) })) };
    v.prompts[0].regions[1].semiAxes[0] = length;
    return M.successRate(v, v.prompts[0], M.train(v, true)[M.ROUNDS], true, "wrong");
  };
  console.log(`wrong answers, region length 0.4 / 0.6: ${pct(ellipse(0.4))} / ${pct(ellipse(0.6))}`);
  assert(Math.abs(ellipse(0.4) - 0.68) < 0.01 && Math.abs(ellipse(0.6) - 0.38) < 0.01, "step 3 quotes 68% and 38%");
  console.log("sliders ok");
}

// step 1 majority vote and step 2 filter threshold
{
  const v = M.VARIANTS.walks;
  const vote = (steps) => M.majorityRates({ ...v, steps }, v.prompts[0], M.zeroWeights(v), [1, 31], 1500);
  const v40 = vote(40);
  const v36 = vote(36);
  console.log(`majority vote k=1/31: 40 steps ${v40.map(pct).join(" ")}, 36 steps ${v36.map(pct).join(" ")}`);
  assert(Math.abs(v40[1] - 0.59) < 0.04, "step 1 quotes about 59% at k = 31");
  assert(v36[1] < v36[0] && v36[1] < 0.04, "voting should make things worse when a wrong answer is most common");

  const am3 = M.VARIANTS.amortize;
  const kept = (keepScale) => {
    const vv = { ...am3, keepScale };
    const w = M.train(vv, true)[M.ROUNDS];
    return vv.prompts.filter((q) => !q.isNew).map((q) => M.successRate(vv, q, w, true));
  };
  const strict = kept(0.5);
  const loose = kept(2.5);
  console.log(`filter 0.5x: ${strict.map(pct).join(" ")}; 2.5x: ${loose.map(pct).join(" ")}`);
  assert(strict.some((x) => x < 0.01) && loose.every((x) => x < 0.4 && x > 0.15), "too strict starves a prompt, too loose copies near-misses");
  console.log("good fits ok");
}
