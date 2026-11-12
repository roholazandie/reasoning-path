// run with: node tools/check-gepa.js
const assert = require("node:assert");
const G = require("../js/gepa-model.js");

const SEEDS = [1, 2, 3, 4, 5, 6];
const GEPA_BUDGET = 6000;
const best = G.testSuccess((t) => [G.NEEDED.H.shift + G.NEEDED.H.tilt * t.height, G.NEEDED.V.shift + G.NEEDED.V.tilt * t.height]);
const level = 0.85 * best;
const median = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
const f = (x) => (Number.isFinite(x) ? String(x) : "never");

console.log(`best possible success ${best.toFixed(2)}, threshold ${level.toFixed(2)}`);
const variants = {
  "gepa + merge": { merge: true },
  gepa: {},
  "greedy selection": { selection: "greedy", merge: true },
  "random edits": { reflective: false, merge: true },
};
const reach = {};
const merges = [];
for (const [name, opt] of Object.entries(variants)) {
  reach[name] = SEEDS.map((seed) => {
    const g = G.createGepa({ ...opt, seed });
    let accepted = 0;
    while (g.rollouts < GEPA_BUDGET) {
      const e = g.iterate();
      if (e.type === "merge" && e.accepted) accepted++;
    }
    if (name === "gepa + merge") merges.push(accepted);
    return G.rolloutsToReach(g.history, level);
  });
  console.log(`${name}: rollouts to threshold ${reach[name].map(f).join(" ")} (median ${f(median(reach[name]))})`);
}
const grpo = SEEDS.map((seed) => G.runGrpo({ seed }));
reach.grpo = grpo.map((r) => G.rolloutsToReach(r.history, level));
console.log(`grpo: rollouts to threshold ${reach.grpo.map(f).join(" ")} (median ${f(median(reach.grpo))}), final ${grpo.map((r) => r.history.at(-1).test.toFixed(2)).join(" ")}`);
console.log(`accepted merges per run: ${merges.join(" ")}`);

assert(median(reach["gepa + merge"]) * 2 < median(reach.grpo), "gepa should need well under half the rollouts of grpo");
assert(median(reach["random edits"]) > median(reach["gepa + merge"]) * 2, "random edits should be much slower than reflection");
assert(grpo.every((r) => r.history.at(-1).test > 0.75), "grpo should get there eventually with enough rollouts");
assert(merges.filter((m) => m > 0).length >= SEEDS.length / 2, "merge should be accepted in most runs");
console.log("ok");

// step 9 quotes these effects of its controls
{
  const run = (opt) => median(SEEDS.map((seed) => G.rolloutsToReach(G.runGepa({ merge: true, ...opt, seed }, GEPA_BUDGET).history, level)));
  const q = [0, 0.35, 0.7, 1].map((quality) => run({ quality }));
  const off = run({ feedback: "binary" });
  const mb1 = run({ minibatch: 1 });
  const mb6 = run({ minibatch: 6 });
  console.log(`quality 0/0.35/0.7/1: ${q.map(f).join(" ")}; text off ${f(off)}; minibatch 1/6: ${f(mb1)} ${f(mb6)}`);
  assert(q[0] > q[1] && q[1] > q[2] && q[2] > q[3], "a better reflector should need fewer rollouts");
  assert(off > 3 * q[2], "pass-or-fail feedback should be much slower than text feedback");
  assert(!Number.isFinite(mb1) && mb6 < q[2], "a minibatch of 1 should stall and 6 should be faster than 3");
  console.log("controls ok");
}
