"""Prepared rollouts for the big half of seminar 7 (optional: the notebook
skips those cells when the files are missing).

Writes one gzipped JSONL per model into ../rollouts/:

    samples_<tag>.jsonl.gz   N answers per problem from base / RL / SFT models
                             of one family (Qwen2.5-Math-1.5B), for pass@k and
                             for counting distinct solutions;
    traces_<tag>.jsonl.gz    a few long reasoning traces per problem with the
                             entropy of the next-token distribution per token.

Every row: {"pid", "question", "gold", "model", "samples": [{"text",
"n_tokens", "entropy"?}]}. Correctness is NOT stored: the notebook recomputes
it from the text with the same checker it uses on live samples.

    python gpu/make_rollouts.py                       # vLLM, defaults below
    python gpu/make_rollouts.py --models sft --n 16   # what the seminar uses for SFT

The seminar's rollouts (13-10-2026): one A100 80 GB shared with other jobs,
--gpu-mem 0.25 (~20 GB), each model in its own process, base and RL with 32
answers, SFT with 16 (its answers are ~3K tokens); ~45 minutes in total.

Memory: one 1.5B model at a time, ~3 GB of bf16 weights plus the KV cache
(~28 KB per token for Qwen2.5-1.5B); ~35 GB keeps ~250 answers of 4K tokens
in flight. Host RAM 10-16 GB (on Slurm ask for --mem=32G), ~10 GB of disk
for the three models in the HF cache.
    python gpu/make_rollouts.py --backend hf --problems 3 --n 4 --models qwen05
                                                      # tiny CPU smoke run

Entropy with vLLM comes from the top-K log-probabilities (K = --topk); the
mass outside the top-K is folded into one bucket, so the value is a lower
bound on the exact entropy. With --backend hf it is exact.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "rollouts"

GSM8K_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/master/"
             "grade_school_math/data/test.jsonl")
MATH500 = "HuggingFaceH4/MATH-500"
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."

MODELS = {
    # tag: (HF id, chat template?, role in the seminar)
    "base": ("Qwen/Qwen2.5-Math-1.5B", False, "samples"),
    "rl": ("sail/Qwen2.5-Math-1.5B-Oat-Zero", False, "samples"),
    "sft": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", True, "samples"),
    "r1trace": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", True, "traces"),
    "qwen05": ("Qwen/Qwen2.5-0.5B-Instruct", True, "samples"),       # smoke tests only
    "qwen05trace": ("Qwen/Qwen2.5-0.5B-Instruct", True, "traces"),   # smoke tests only
}
# Qwen2.5-Math base and its R1-Zero-style RL descendant are prompted the way
# the Oat-Zero recipe trains them (Qwen-Math template, no system chat turn).
QWEN_MATH_TEMPLATE = ("<|im_start|>system\nPlease reason step by step, and put your final "
                      "answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n"
                      "<|im_start|>assistant\n")


def integer_answer(s: str) -> float | None:
    s = s.replace(",", "").replace("$", "").strip()
    try:
        v = float(s)
    except ValueError:
        return None
    return v if v == int(v) else None


def load_problems(source: str, k: int, seed: int) -> list[dict]:
    if source == "gsm8k":
        rows = [json.loads(l) for l in urllib.request.urlopen(GSM8K_URL).read().decode().splitlines()]
        probs = [{"pid": f"gsm8k-{i}", "question": r["question"],
                  "gold": float(r["answer"].split("####")[-1].strip().replace(",", ""))}
                 for i, r in enumerate(rows)]
    else:
        from datasets import load_dataset
        ds = load_dataset(MATH500, split="test")
        probs = []
        for r in ds:
            g = integer_answer(r["answer"])
            if g is not None:
                probs.append({"pid": f"math500-{r['unique_id']}", "question": r["problem"],
                              "gold": g, "level": r["level"]})
    random.Random(seed).shuffle(probs)
    return probs[:k]


def build_prompt(tok, tag: str, q: str) -> str:
    hf_id, chat, _ = MODELS[tag]
    if not chat:
        return QWEN_MATH_TEMPLATE.format(q=q)
    msgs = [{"role": "user", "content": q + "\n" + SYSTEM}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def entropy_from_topk(logprobs: dict) -> float:
    ps = [math.exp(lp.logprob) for lp in logprobs.values()]
    rest = max(0.0, 1.0 - sum(ps))
    return -sum(p * math.log(p) for p in ps if p > 0) - (rest * math.log(rest) if rest > 0 else 0.0)


def run_vllm(tag, problems, n, max_tokens, temperature, topk, with_entropy, seed, gpu_mem):
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    hf_id = MODELS[tag][0]
    # Qwen2.5-Math has a 4K context: cap the answer so that prompt + answer fit
    ctx = min(max_tokens + 1024, AutoConfig.from_pretrained(hf_id).max_position_embeddings)
    max_tokens = min(max_tokens, ctx - 1024)
    llm = LLM(hf_id, dtype="bfloat16", gpu_memory_utilization=gpu_mem, seed=seed,
              max_model_len=ctx)
    tok = llm.get_tokenizer()
    prompts = [build_prompt(tok, tag, p["question"]) for p in problems]
    # n=1 per request: n>1 in LLM.generate silently aborts attempts in some versions
    reqs = [pr for pr in prompts for _ in range(n)]
    # one seed per request: with a shared seed, identical prompts give identical answers
    sps = [SamplingParams(n=1, temperature=temperature, top_p=1.0, max_tokens=max_tokens,
                          logprobs=topk if with_entropy else None, seed=seed * 1_000_003 + j)
           for j in range(len(reqs))]
    outs = llm.generate(reqs, sps)
    rows = []
    for i, p in enumerate(problems):
        samples = []
        for o in outs[i * n:(i + 1) * n]:
            c = o.outputs[0]
            if c.finish_reason == "abort":
                raise RuntimeError(f"{tag}: aborted sample for {p['pid']}")
            s = {"text": c.text, "n_tokens": len(c.token_ids), "finish": c.finish_reason}
            if with_entropy:
                s["tokens"] = [tok.decode([t]) for t in c.token_ids]
                s["entropy"] = [round(entropy_from_topk(lp), 4) for lp in c.logprobs]
            samples.append(s)
        rows.append({**p, "model": hf_id, "samples": samples})
    return rows


def run_hf(tag, problems, n, max_tokens, temperature, with_entropy, seed):
    import sys
    sys.path.insert(0, str(HERE.parent))
    import torch
    from rl_lab import load_model
    hf_id = MODELS[tag][0]
    tok, model = load_model(hf_id)
    rows = []
    for p in problems:
        enc = tok([build_prompt(tok, tag, p["question"])] * n, return_tensors="pt",
                  padding=True).to(model.device)
        torch.manual_seed(seed)
        out = model.generate(**enc, do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
                             max_new_tokens=max_tokens, pad_token_id=tok.pad_token_id,
                             output_logits=with_entropy, return_dict_in_generate=True)
        gen = out.sequences[:, enc.input_ids.shape[1]:].cpu()
        samples = []
        for j, row in enumerate(gen):
            ids = row[row != tok.pad_token_id]
            s = {"text": tok.decode(ids, skip_special_tokens=True), "n_tokens": int(ids.numel())}
            if with_entropy:
                ent = []
                for lg in out.logits[:len(ids)]:
                    lp = torch.log_softmax(lg[j].float(), -1)
                    ent.append(round(float(-(lp.exp() * lp).sum()), 4))
                s["tokens"] = [tok.decode([t]) for t in ids.tolist()]
                s["entropy"] = ent
            samples.append(s)
        rows.append({**p, "model": hf_id, "samples": samples})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--source", choices=["math500", "gsm8k"], default="math500")
    ap.add_argument("--models", nargs="+", default=["base", "rl", "sft", "r1trace"])
    ap.add_argument("--problems", type=int, default=60)
    ap.add_argument("--n", type=int, default=32, help="answers per problem (samples)")
    ap.add_argument("--trace-problems", type=int, default=20)
    ap.add_argument("--trace-n", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--trace-max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--gpu-mem", type=float, default=0.25,
                    help="vLLM share of GPU memory: 0.25 of an H200 is ~35 GB, enough for "
                         "~1M tokens of KV cache of a 1.5B model (~28 KB per token)")
    a = ap.parse_args()

    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    problems = load_problems(a.source, max(a.problems, a.trace_problems), a.seed)
    for tag in a.models:
        traces = MODELS[tag][2] == "traces"
        probs = problems[:a.trace_problems] if traces else problems[:a.problems]
        n = a.trace_n if traces else a.n
        mt = a.trace_max_tokens if traces else a.max_tokens
        if a.backend == "vllm":
            rows = run_vllm(tag, probs, n, mt, a.temperature, a.topk, traces, a.seed, a.gpu_mem)
        else:
            rows = run_hf(tag, probs, n, mt, a.temperature, traces, a.seed)
        name = ("traces_" if traces else "samples_") + tag + ".jsonl.gz"
        with gzip.open(out / name, "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} problems x {n}")


if __name__ == "__main__":
    main()
