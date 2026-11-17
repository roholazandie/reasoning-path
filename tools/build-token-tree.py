"""Build the sampled token paths for the real-token-paths step from a local model.

run with: <python with torch + transformers> tools/build-token-tree.py [--probe] [--ids ID ...]
"""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B"
SYSTEM = (
    "Solve the problem. Think in at most three short sentences, "
    "then end with a final line of the form 'Answer: <number>'."
)
# the policy is the model at this temperature with no top-k or top-p; every probability shown is from it
TEMPERATURE = 0.7
TOP_ALTS = 4
MAX_NEW = 100
TREE_PATHS = 16
SAMPLES = 400
SEED = 7
OUT = Path(__file__).resolve().parent.parent / "data" / "token-trees.json"

CANDIDATES = [
    {"id": "multiply", "question": "What is 37 times 48?", "answer": 1776},
    {"id": "percent", "question": "What is 23% of 460?", "answer": 105.8},
    {"id": "bat-ball", "question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How many cents does the ball cost?", "answer": 5},
    {"id": "strawberry", "question": "How many times does the letter r appear in the word strawberry?", "answer": 3},
    {"id": "days", "question": "If today is Wednesday, what day of the week is it 100 days from now? Give the answer as a number where Monday is 1 and Sunday is 7.", "answer": 5},
    {"id": "sheep", "question": "A farmer has 17 sheep. All but 9 run away. How many sheep are left?", "answer": 9},
    {"id": "primes", "question": "How many prime numbers are there between 50 and 100?", "answer": 10},
    {"id": "divide", "question": "What is 1001 divided by 7, minus 13?", "answer": 130},
    {"id": "widgets", "question": "If 5 machines make 5 widgets in 5 minutes, how many minutes do 100 machines take to make 100 widgets?", "answer": 5},
    {"id": "square", "question": "What is the sum of the squares of 7, 8 and 9?", "answer": 194},
]

ANSWER_RE = re.compile(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)")


def parse_answer(text):
    """Return the number after the first 'Answer:' in text, or None."""
    found = ANSWER_RE.search(text)
    if not found:
        return None
    try:
        return float(found.group(1).replace(",", ""))
    except ValueError:
        return None


def answer_done(text):
    """Return whether text holds a first answer whose number can no longer grow."""
    found = ANSWER_RE.search(text)
    if not found or found.end() >= len(text):
        return False
    rest = text[found.end():]
    if rest[0].isdigit():
        return False
    # a trailing comma or point may still be followed by more digits
    return not (rest[0] in ",." and (len(rest) == 1 or rest[1].isdigit()))


def is_correct(value, answer):
    """Return whether a parsed answer matches the reference."""
    return value is not None and abs(value - answer) < 1e-6


def significant(p):
    """Round a probability to four significant digits, so a rare token never rounds to zero."""
    return float(f"{p:.4g}")


def answer_key(value):
    """Return a short label for a parsed answer."""
    if value is None:
        return "none"
    return str(int(value)) if float(value).is_integer() else str(value)


class Model:
    """Wrap the tokenizer and model with the few calls the builder needs."""

    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
        self.lm.eval()
        self.stop = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>"),
                     self.tok.convert_tokens_to_ids("<|endoftext|>")}

    def prompt_ids(self, question):
        """Return the chat-formatted prompt token ids for a question."""
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tok(text, return_tensors="pt").input_ids[0].tolist()

    def trim(self, ids):
        """Cut a generated id list at its first stop token or right after its first complete answer."""
        for k, t in enumerate(ids):
            if t in self.stop:
                return ids[:k + 1]
            # the verifier reads only the first answer, so what follows it is never scored or shown
            if answer_done(self.tok.decode(ids[:k + 1])):
                before = parse_answer(self.tok.decode(ids[:k]))
                return ids[:k] if before == parse_answer(self.tok.decode(ids[:k + 1])) else ids[:k + 1]
        return ids

    @torch.no_grad()
    def sample(self, prefix, n, seed):
        """Sample n continuations of prefix from the policy and return their id lists."""
        torch.manual_seed(seed)
        out = []
        for start in range(0, n, 100):
            size = min(100, n - start)
            batch = torch.tensor([prefix] * size, device="cuda")
            gen = self.lm.generate(
                batch, attention_mask=torch.ones_like(batch), do_sample=True, temperature=TEMPERATURE,
                top_k=0, top_p=1.0, max_new_tokens=MAX_NEW, pad_token_id=self.tok.eos_token_id,
            )
            out += [self.trim(row.tolist()) for row in gen[:, batch.shape[1]:]]
        return out

    @torch.no_grad()
    def annotate(self, prefix, ids):
        """Return each id of a continuation with its policy probability and top alternatives."""
        full = torch.tensor([prefix + ids], device="cuda")
        logits = self.lm(input_ids=full).logits[0, len(prefix) - 1:-1].float()
        probs = torch.softmax(logits / TEMPERATURE, dim=-1)
        rows = []
        for k, token in enumerate(ids):
            top = torch.topk(probs[k], TOP_ALTS + 1)
            alts = [(int(i), float(v)) for i, v in zip(top.indices, top.values) if int(i) != token][:TOP_ALTS - 1]
            rows.append({"id": token, "p": float(probs[k, token]), "alts": alts})
        return rows

    def text(self, ids):
        """Return the visible text of an id list without stop tokens."""
        return self.tok.decode([t for t in ids if t not in self.stop])

    def piece(self, token):
        """Return the visible text of one token."""
        return self.tok.decode([token])


def probe(model):
    """Print the sampled success rate of every candidate prompt."""
    for c in CANDIDATES:
        runs = model.sample(model.prompt_ids(c["question"]), 200, SEED)
        values = [parse_answer(model.text(r)) for r in runs]
        hits = sum(is_correct(v, c["answer"]) for v in values)
        none = sum(v is None for v in values)
        print(f"{c['id']:<11} p_x ~ {hits / len(values):.2f}  no answer {none / len(values):.2f}")


def export_path(model, c, rows, extra, prefix=()):
    """Return one path as page data: tokens, answer and whether it lands; prefix is the shared start of a what-if."""
    value = parse_answer(model.text(list(prefix) + [r["id"] for r in rows]))
    return {
        **extra,
        "tokens": [
            {"t": model.piece(r["id"]), "p": significant(r["p"]),
             "alts": [[model.piece(a), significant(v)] for a, v in r["alts"]]}
            for r in rows if r["id"] not in model.stop
        ],
        "answer": value,
        "correct": is_correct(value, c["answer"]),
    }


def build_prompt(model, c):
    """Sample the success estimate, the tree paths and one what-if branch per tree path."""
    base = model.prompt_ids(c["question"])
    runs = model.sample(base, SAMPLES, SEED)
    values = [parse_answer(model.text(r)) for r in runs]
    counts = {}
    for v in values:
        counts[answer_key(v)] = counts.get(answer_key(v), 0) + 1
    hits = sum(is_correct(v, c["answer"]) for v in values)

    paths = []
    # the tree shows the first TREE_PATHS of the same samples, so its share of landing paths matches the estimate
    for i, ids in enumerate(runs[:TREE_PATHS]):
        rows = model.annotate(base, ids)
        paths.append(export_path(model, c, rows, {"kind": "sample", "parent": None, "forkAt": 0}))
        # the what-if forks at the most uncertain visible token and forces its most likely alternative
        # a fork after the answer has started changes little, so only tokens before it qualify
        def forkable(k):
            r = rows[k]
            open_alts = [a for a in r["alts"] if a[0] not in model.stop]
            return r["id"] not in model.stop and open_alts and not ANSWER_RE.search(model.text(ids[:k]))
        visible = [k for k in range(len(rows)) if forkable(k)]
        if not visible:
            continue
        k = min(visible, key=lambda j: rows[j]["p"])
        forced = next(a for a, _ in rows[k]["alts"] if a not in model.stop)
        head = ids[:k] + [forced]
        tail = model.sample(base + head, 1, SEED + 1000 * i + k)[0]
        what_if = model.annotate(base, head + tail)[k:]
        paths.append(export_path(model, c, what_if, {"kind": "whatif", "parent": len(paths) - 1, "forkAt": k}, ids[:k]))
    return {**c, "paths": paths, "samples": {"n": len(values), "hits": hits, "answers": counts}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--ids", nargs="*", default=None)
    args = parser.parse_args()
    model = Model()
    if args.probe:
        probe(model)
        return
    chosen = [c for c in CANDIDATES if args.ids is None or c["id"] in args.ids]
    prompts = []
    for c in chosen:
        prompts.append(build_prompt(model, c))
        s = prompts[-1]["samples"]
        print(f"{c['id']}: sampled p_x {s['hits'] / s['n']:.2f}")
    OUT.parent.mkdir(exist_ok=True)
    meta = {"model": MODEL, "system": SYSTEM, "temperature": TEMPERATURE, "seed": SEED}
    OUT.write_text(json.dumps({**meta, "prompts": prompts}, ensure_ascii=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
