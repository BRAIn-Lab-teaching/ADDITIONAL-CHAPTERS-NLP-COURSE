"""Small toolbox for seminar 7: loading models, sampling answers with their
token log-probabilities, checking answers, reading the optional big rollouts.

Everything the seminar is about (advantages, the GRPO loss, pass@k, where the
answer appears in a trace) is written in the notebook itself; this file only
holds the plumbing so that the notebook cells stay short.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import torch

__all__ = [
    "POLICY", "THINKER", "DEVICE", "DTYPE", "SYSTEM_PROMPT", "Sample", "load_model", "chat_prompt",
    "sample_group", "completion_logprobs", "generate_trace", "extract_answer",
    "is_correct", "split_sentences", "load_rollouts", "sentence_tree",
]

POLICY = "Qwen/Qwen2.5-0.5B-Instruct"     # parts 1, 2, 4: a small instruct model
THINKER = "Qwen/Qwen3-0.6B"               # part 3: a small model with a thinking mode

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# fp32 everywhere, GPU included. A T4 (Colab) has no bf16, Qwen activations can
# overflow in fp16, and the Adam step of part 1 (lr 1e-6) is lost to rounding in
# any 16-bit format. Two models under 0.6B take ~5 GB in fp32.
DTYPE = torch.float32
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", message="IProgress not found")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

_CACHE: dict[str, tuple] = {}


def load_model(name: str):
    """(tokenizer, model) on DEVICE, loaded once per session."""
    if name not in _CACHE:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging as hf_logging
        from huggingface_hub.utils import logging as hub_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        hub_logging.set_verbosity_error()
        tok = AutoTokenizer.from_pretrained(name)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        import transformers
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"   # renamed in 4.56
        model = AutoModelForCausalLM.from_pretrained(name, **{key: DTYPE}).to(DEVICE)
        model.eval()
        _CACHE[name] = (tok, model)
    return _CACHE[name]


def chat_prompt(tok, question: str, *, system: str | None = SYSTEM_PROMPT,
                thinking: bool | None = None) -> str:
    msgs = ([{"role": "system", "content": system}] if system else [])
    msgs.append({"role": "user", "content": question})
    kw = {} if thinking is None else {"enable_thinking": thinking}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)


# ------------------------------------------------------------- answers ---
_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_answer(text: str) -> float | None:
    """The last \\boxed{...} if there is one, else the last number in the text."""
    boxed = _BOXED.findall(text)
    cand = boxed[-1] if boxed else (_NUMBER.findall(text) or [""])[-1]
    cand = cand.replace(",", "").replace("$", "").replace("\\%", "").strip()
    m = _NUMBER.search(cand)
    if not m:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None


def is_correct(text: str, gold: float) -> bool:
    a = extract_answer(text)
    return a is not None and math.isclose(a, float(gold), rel_tol=0, abs_tol=1e-6)


# ------------------------------------------------------------ sampling ---
@dataclass
class Sample:
    text: str
    prompt_ids: torch.Tensor          # (P,)
    completion_ids: torch.Tensor      # (T,) without padding
    correct: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        return int(self.completion_ids.numel())


_SAMPLES: dict[tuple, list] = {}


@torch.no_grad()
def sample_group(name: str, question: str, gold: float, *, G: int = 8,
                 max_new_tokens: int = 512, temperature: float = 1.0,
                 seed: int = 0, system: str | None = SYSTEM_PROMPT) -> list[Sample]:
    """G independent answers to one question, each checked against gold.
    The same call twice in one session returns the same samples without
    generating again."""
    key = (name, question, G, max_new_tokens, temperature, seed, system)
    if key in _SAMPLES:
        return [Sample(s.text, s.prompt_ids, s.completion_ids, is_correct(s.text, gold))
                for s in _SAMPLES[key]]
    tok, model = load_model(name)
    prompt = chat_prompt(tok, question, system=system)
    enc = tok([prompt] * G, return_tensors="pt", padding=True).to(DEVICE)
    torch.manual_seed(seed)
    out = model.generate(**enc, do_sample=True, temperature=temperature, top_p=1.0,
                         top_k=0, max_new_tokens=max_new_tokens,
                         pad_token_id=tok.pad_token_id)
    P = enc.input_ids.shape[1]
    prompt_ids = enc.input_ids[0][enc.attention_mask[0].bool()].cpu()
    stop = {tok.eos_token_id, tok.pad_token_id}
    samples = []
    for row in out[:, P:].cpu():
        row = row.tolist()
        end = next((i for i, t in enumerate(row) if t in stop), len(row) - 1)
        ids = torch.tensor(row[:end + 1])            # up to and including the stop token
        text = tok.decode(ids, skip_special_tokens=True)
        samples.append(Sample(text, prompt_ids, ids, is_correct(text, gold)))
    _SAMPLES[key] = samples
    return samples


def completion_logprobs(model, samples: list[Sample], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log p(o_t | q, o_<t) for every sample, padded on the right.
    Returns (logp, mask), both (G, T_max); gradients flow through logp.

    One sample at a time, and the log-softmax only over the completion: the
    vocabulary has ~150K entries, so logits of a whole group in fp32 take
    several GB and do not fit a T4 next to the models."""
    T = max(s.n_tokens for s in samples)
    rows = []
    for s in samples:
        ids = torch.cat([s.prompt_ids, s.completion_ids])[None].to(DEVICE)
        p, n = len(s.prompt_ids), s.n_tokens
        logits = model(input_ids=ids).logits[0, p - 1:p - 1 + n].float()
        lp = torch.log_softmax(logits, -1).gather(-1, ids[0, p:p + n, None]).squeeze(-1)
        rows.append(torch.nn.functional.pad(lp, (0, T - n)))
    logp = torch.stack(rows)
    mask = torch.zeros((len(samples), T), device=DEVICE)
    for i, s in enumerate(samples):
        mask[i, :s.n_tokens] = 1
    return logp, mask


@torch.no_grad()
def generate_trace(name: str, question: str, *, max_new_tokens: int = 1500,
                   temperature: float = 0.6, seed: int = 0,
                   thinking: bool | None = True) -> dict:
    """One answer, token by token, with the entropy of the full next-token
    distribution at every step (from the raw logits, before sampling filters)."""
    tok, model = load_model(name)
    prompt = chat_prompt(tok, question, system=None, thinking=thinking)
    enc = tok(prompt, return_tensors="pt").to(DEVICE)
    torch.manual_seed(seed)
    out = model.generate(**enc, do_sample=True, temperature=temperature, top_p=0.95,
                         max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                         output_logits=True, return_dict_in_generate=True)
    ids = out.sequences[0, enc.input_ids.shape[1]:].cpu()
    ent = []
    for lg in out.logits:
        lp = torch.log_softmax(lg[0].float(), -1)
        ent.append(float(-(lp.exp() * lp).sum()))
    pieces = [tok.decode(ids[:i + 1]) for i in range(len(ids))]
    tokens = [pieces[0]] + [pieces[i][len(pieces[i - 1]):] for i in range(1, len(pieces))]
    return {"ids": ids.tolist(), "tokens": tokens, "entropy": ent[:len(tokens)],
            "text": tok.decode(ids, skip_special_tokens=True)}


# ------------------------------------------------------- trace helpers ---
_SENT_END = re.compile(r"(?<=[^\d][.!?])\s+|\n+")      # not after "1." in a numbered list


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_END.split(text) if s and s.strip()]


_LIST_MARK = re.compile(r"^(?:[-*#>\s]+|\d+[.)]\s*|step\s*\d+\s*[:.]\s*)+")
_WORD = re.compile(r"[a-z]+|\d+(?:\.\d+)?")


def _sentence_key(s: str, n_words: int) -> str:
    """The first n_words words of a sentence, lowercased, without list
    numbering and markdown, so that "1. **Calculate the total cost of gas**"
    and "Calculate the total cost without..." share a branch."""
    words = _WORD.findall(_LIST_MARK.sub("", s.lower()))
    return " ".join(words[:n_words])


def sentence_tree(texts: list[str], labels: list[str], max_depth: int = 6,
                  width: int = 60, n_words: int | None = None) -> dict:
    """Prefix tree over sentences: answers whose sentences open with the same
    n_words words (None: the same first `width` characters) share a branch.
    Returns plotly icicle/sunburst columns."""
    ids, parents, names, values, colors = ["root"], [""], ["question"], [len(texts)], ["all"]
    counts: dict[str, int] = {}
    owner: dict[str, set] = {}
    for text, lab in zip(texts, labels):
        sents = split_sentences(text)[:max_depth]
        path = "root"
        for d, s in enumerate(sents):
            key = (_sentence_key(s, n_words) if n_words
                   else re.sub(r"\s+", " ", s.lower())[:width])
            node = f"{path}/{d}:{key}"
            if node not in counts:
                ids.append(node)
                parents.append(path)
                names.append(s[:width] + ("…" if len(s) > width else ""))
                counts[node] = 0
                owner[node] = set()
            counts[node] += 1
            owner[node].add(lab)
            path = node
    for k, n in enumerate(ids[1:], 1):
        values.append(counts[n])
        o = owner[n]
        colors.append(next(iter(o)) if len(o) == 1 else "mixed")
        if n_words and counts[n] > 1:      # a merged branch: label it by the shared words
            names[k] = n.rsplit(":", 1)[1] + " …"
    return {"ids": ids, "parents": parents, "names": names, "values": values,
            "labels": colors}


# -------------------------------------------------- prepared rollouts ---
def load_rollouts(path: str | Path) -> list[dict] | None:
    """Rows written by gpu/make_rollouts.py, or None if the file is absent."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        if f.read(40).startswith(b"version https://git-lfs"):
            print(f"{path.name} is a Git LFS pointer, not the data: run `git lfs pull`")
            return None
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
