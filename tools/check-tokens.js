// run with: node tools/check-tokens.js
// checks the token-path data behind step 2 and the claims the page makes about it.
const assert = require("node:assert");
const data = require("../data/token-trees.json");

const ANSWER = /Answer:\s*\$?\s*(-?[\d,]*\.?\d+)/;
const byId = Object.fromEntries(data.prompts.map((p) => [p.id, p]));
assert.deepStrictEqual(Object.keys(byId).sort(), ["multiply", "sheep", "square", "strawberry", "widgets"], "the page has a button for each of these prompts");

for (const p of data.prompts) {
  const samples = p.paths.filter((q) => q.kind === "sample");
  assert.strictEqual(samples.length, 16, `${p.id}: the page says sixteen paths are drawn`);
  for (const [i, q] of p.paths.entries()) {
    const text = q.kind === "sample" ? q.tokens.map((t) => t.t).join("")
      : p.paths[q.parent].tokens.slice(0, q.forkAt).concat(q.tokens).map((t) => t.t).join("");
    const m = text.match(ANSWER);
    const value = m ? Number(m[1].replace(/,/g, "")) : null;
    assert.strictEqual(value, q.answer, `${p.id} path ${i}: stored answer matches its text`);
    assert.strictEqual(q.correct, value !== null && Math.abs(value - p.answer) < 1e-6, `${p.id} path ${i}: correct flag matches the reference`);
    for (const t of q.tokens) {
      assert(t.p > 0 && t.p <= 1, `${p.id} path ${i}: token probabilities lie in (0, 1]`);
      assert(t.alts.every(([, a]) => a + t.p <= 1 + 1e-4), `${p.id} path ${i}: a token and an alternative never sum past 1`);
    }
    if (q.kind === "whatif") {
      const parent = p.paths[q.parent];
      assert(parent.kind === "sample" && q.forkAt < parent.tokens.length, `${p.id} path ${i}: forks inside its parent`);
      const fork = parent.tokens[q.forkAt];
      // the forced token is the parent's most likely alternative, at its least likely token before the answer is complete
      const isStop = (t) => /^<\|.*\|>$/.test(t);
      assert.strictEqual(q.tokens[0].t, fork.alts.find(([t]) => !isStop(t))[0], `${p.id} path ${i}: forces the top alternative that is not a stop token`);
      const open = parent.tokens.filter((t, k) => !ANSWER.test(parent.tokens.slice(0, k).map((x) => x.t).join("")) && t.alts.some(([a]) => !isStop(a)));
      assert(open.includes(fork), `${p.id} path ${i}: forks before the answer starts`);
      assert(open.every((t) => t.p >= fork.p), `${p.id} path ${i}: forks at the least likely token before the answer`);
    }
  }
  const s = p.samples;
  assert.strictEqual(Object.values(s.answers).reduce((a, b) => a + b, 0), s.n, `${p.id}: answer counts add up to the samples`);
  assert.strictEqual(s.answers[String(p.answer)] || 0, s.hits, `${p.id}: hits equal the count of the correct answer`);
  const best = Math.max(...samples.map((q) => q.tokens.reduce((a, t) => a * t.p, 1)));
  console.log(`${p.id}: p_x ${(s.hits / s.n).toFixed(2)}, drawn ${samples.filter((q) => q.correct).length}/16, most likely drawn path ${best.toFixed(3)}`);
  const landing = new Set(samples.filter((q) => q.correct).map((q) => q.tokens.map((t) => t.t).join("")));
  const landed = samples.filter((q) => q.correct).length;
  console.log(`  ${landing.size} distinct texts among ${landed} landing paths`);
  assert(landing.size >= landed / 2, `${p.id}: the page says the landing paths are many different texts`);
  if (p.id === "sheep") assert(Math.abs(best - 0.19) < 0.01, "sheep: the page quotes about 19% for the top path");
}

// the strawberry paragraph: the most common answer is wrong, so a vote locks onto it
const straw = byId.strawberry.samples.answers;
assert(straw["2"] > straw["3"], "strawberry: 2 is more common than 3");
console.log("ok");
