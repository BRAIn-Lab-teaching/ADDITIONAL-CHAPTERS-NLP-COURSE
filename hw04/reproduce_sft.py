"""Reproduce the prepared SFT adapter and its answers on the fixed test set."""
import json
import os
from importlib import metadata
from pathlib import Path

import torch
from packaging.version import Version
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from privacy_data import make_examples

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def main():
    try:
        if Version(metadata.version("torchao")) < Version("0.16"):
            from peft.tuners.lora import torchao as dispatch
            dispatch.is_torchao_available = lambda: False
    except metadata.PackageNotFoundError:
        pass
    assert torch.cuda.is_available(), "GPU required"
    torch.manual_seed(42)
    out = Path(os.environ.get("HW04_OUTPUT", "/content/hw04_sft")); out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((Path(__file__).parent / "assets/sft_baseline.json").read_text(encoding="utf-8"))["rows"]
    train = make_examples("train", 200)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32)
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")).to("cuda")
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=5e-5)
    model.train()
    for step in range(316):
        optim.zero_grad()
        for row in (train[(2*step+i) % len(train)] for i in range(2)):
            messages = [{"role":"user", "content":row["prompt"]}]
            prefix = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            whole = tok.apply_chat_template(messages + [{"role":"assistant", "content":row["chosen"]}], tokenize=False)
            pre = tok(prefix, add_special_tokens=False)["input_ids"]
            ids = tok(whole, add_special_tokens=False)["input_ids"]
            assert ids[:len(pre)] == pre
            x = torch.tensor([ids], device="cuda"); labels=x.clone(); labels[:,:len(pre)]=-100
            (model(input_ids=x, labels=labels).loss/2).backward()
        optim.step()
    model.save_pretrained(out/"sft_adapter")

    def generate(prompts):
        outputs = []
        model.eval()
        for i in range(0, len(prompts), 8):
            text = [tok.apply_chat_template([{"role":"user", "content":p}], tokenize=False, add_generation_prompt=True)
                    for p in prompts[i:i+8]]
            inputs = tok(text, padding=True, add_special_tokens=False, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                ids = model.generate(**inputs, do_sample=False, max_new_tokens=48, pad_token_id=tok.eos_token_id)
            for tail in ids[:, inputs["input_ids"].shape[1]:]:
                eos = (tail == tok.eos_token_id).nonzero(); ended = bool(len(eos))
                if ended: tail = tail[:int(eos[0])+1]
                outputs.append((tok.decode(tail, skip_special_tokens=True), ended))
        return outputs

    answers = generate([r["prompt"] for r in rows])
    indices = [i for i,r in enumerate(rows) if r["control"] is not None]
    controls = generate([rows[i]["control"] for i in indices])
    records = [{**r, "output":text, "ended":ended, "control_output":None, "control_ended":None}
               for r, (text, ended) in zip(rows, answers, strict=True)]
    for i, (text, ended) in zip(indices, controls, strict=True):
        records[i]["control_output"], records[i]["control_ended"] = text, ended
    (out/"sft_outputs.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} answers and the SFT adapter to {out}")


if __name__ == "__main__": main()
