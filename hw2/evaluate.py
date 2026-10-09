"""Evaluate BF16, SeedLM-compressed, RTN- or AWQ-quantized weights."""
import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from compress import linear_layers
from seedlm import Compressed, Config, decompress

ZS_TASKS = ["arc_easy", "arc_challenge", "hellaswag", "winogrande", "boolq"]


@torch.no_grad()
def rtn(W: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """Round-to-nearest, asymmetric min-max, per output channel (group<=0) or per group."""
    out_f, in_f = W.shape
    g = in_f if group <= 0 else group
    assert in_f % g == 0
    x = W.float().reshape(out_f, in_f // g, g)
    lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    scale = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-8)
    zp = torch.round(-lo / scale)
    q = (torch.round(x / scale) + zp).clamp(0, 2 ** bits - 1)
    return ((q - zp) * scale).reshape(out_f, in_f)


def apply_seedlm(model, cdir, dev):
    meta = json.load(open(os.path.join(cdir, "meta.json")))
    cfg = Config(meta["C"], meta["P"], meta["K"])
    n = 0
    for name, mod in linear_layers(model):
        path = os.path.join(cdir, "layers", name + ".pt")
        if not os.path.exists(path):
            continue
        d = torch.load(path)
        c = Compressed(tuple(d["shape"]), cfg, d["seeds"], d["exps"], d["q"])
        w = decompress(c, dev)
        mod.weight.data.copy_(w.to(mod.weight.dtype))
        n += 1
    print(f"decoded {n} SeedLM layers ({cfg.bits_per_weight:.2f} bits/weight)")
    return cfg.bits_per_weight


def apply_rtn(model, bits, group, dev):
    for name, mod in linear_layers(model):
        mod.weight.data.copy_(rtn(mod.weight.data.to(dev), bits, group).to(mod.weight.dtype))
    return bits + (16 + 16) / group if group > 0 else bits


@torch.no_grad()
def wikitext2_ppl(model, tok, seqlen, max_windows=None):
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids
    n = ids.shape[1] // seqlen
    if max_windows:
        n = min(n, max_windows)
    dev = next(model.parameters()).device
    nll = 0.0
    for i in range(n):
        x = ids[:, i * seqlen:(i + 1) * seqlen].to(dev)
        logits = model(x).logits.float()
        loss = torch.nn.functional.cross_entropy(logits[0, :-1], x[0, 1:].to(logits.device), reduction="sum")
        nll += loss.item()
    return float(torch.exp(torch.tensor(nll / (n * (seqlen - 1))))), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", choices=["fp16", "seedlm", "rtn", "awq", "omniquant", "quip"], required=True)
    ap.add_argument("--compressed", help="dir produced by compress.py (method=seedlm)")
    ap.add_argument("--bits", type=int, default=4, help="RTN/AWQ bits")
    ap.add_argument("--group", type=int, default=0, help="RTN/AWQ group size, 0 = per channel")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--zeroshot", action="store_true")
    ap.add_argument("--zs-batch", type=int, default=16)
    ap.add_argument("--zs-limit", type=int, default=None, help="examples per task (debug)")
    ap.add_argument("--calib", choices=["pile", "wikitext"], default="pile", help="AWQ calibration data")
    ap.add_argument("--awq-samples", type=int, default=128)
    ap.add_argument("--awq-tokens", type=int, default=32768, help="tokens per input used for AWQ search")
    ap.add_argument("--omni-epochs", type=int, default=20)
    ap.add_argument("--omni-samples", type=int, default=128)
    ap.add_argument("--quip-repo", default="third_party/quip-sharp")
    ap.add_argument("--quip-samples", type=int, default=128)
    ap.add_argument("--quip-tune-iters", type=int, default=10, help="extra LDLQ refinement passes (official: 10); fewer = faster")
    ap.add_argument("--quip-ckpt", default=None, help="dir for per-layer checkpoints (default runs/quip_<model>_w<bits>)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    assert torch.cuda.is_available(), "evaluate.py runs on a GPU only (no CPU fallback)"
    dev = a.device
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                 device_map={"": dev}).eval()
    bpw = 16.0
    T_START = time.time()
    if a.method == "seedlm":
        bpw = apply_seedlm(model, a.compressed, dev)
    elif a.method == "rtn":
        bpw = apply_rtn(model, a.bits, a.group, dev)
    elif a.method == "omniquant":
        from omniquant import omniquant
        t0 = time.time()
        omniquant(model, tok, a.bits, a.group, a.omni_samples, 2048, a.omni_epochs, device=dev)
        print(f"OmniQuant took {time.time() - t0:.0f}s", flush=True)
        bpw = a.bits + (32 / a.group if a.group > 0 else 0)
    elif a.method == "quip":
        from quip_baseline import quip_quantize
        t0 = time.time()
        ckpt = a.quip_ckpt or f"runs/quip_{a.model.split('/')[-1]}_w{a.bits}_t{a.quip_tune_iters}"
        quip_quantize(model, tok, a.bits, a.quip_repo, a.quip_samples, 2048, a.calib,
                      tune_iters=a.quip_tune_iters, device=dev, ckpt_dir=ckpt)
        print(f"QuIP# took {time.time() - t0:.0f}s", flush=True)
        bpw = float(a.bits)
    elif a.method == "awq":
        from awq import awq_quantize
        t0 = time.time()
        awq_quantize(model, tok, a.bits, a.group, a.awq_samples, 512, a.calib, a.awq_tokens, dev)
        print(f"AWQ quantization took {time.time() - t0:.0f}s", flush=True)
        bpw = a.bits + (32 / a.group if a.group > 0 else 0)

    res = dict(model=a.model, method=a.method, bits_per_weight=bpw, compressed=a.compressed,
               rtn_bits=a.bits if a.method in ("rtn", "awq", "omniquant", "quip") else None,
               rtn_group=a.group if a.method in ("rtn", "awq", "omniquant") else None,
               calib=a.calib if a.method in ("awq", "quip") else None,
               quip_tune_iters=a.quip_tune_iters if a.method == "quip" else None,
               omni_epochs=a.omni_epochs if a.method == "omniquant" else None)
    res["quant_seconds"] = round(time.time() - T_START)
    t0 = time.time()
    res["wikitext2_ppl"], res["ppl_windows"] = wikitext2_ppl(model, tok, a.seqlen, a.max_windows)
    print(f"WikiText-2 ppl = {res['wikitext2_ppl']:.3f} ({res['ppl_windows']} x {a.seqlen} tokens, "
          f"{time.time() - t0:.0f}s)", flush=True)

    if a.zeroshot:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=a.zs_batch)
        r = lm_eval.simple_evaluate(model=lm, tasks=ZS_TASKS, limit=a.zs_limit)["results"]
        res["zero_shot"] = {t: r[t].get("acc,none") for t in ZS_TASKS}
        res["zero_shot_mean"] = sum(res["zero_shot"].values()) / len(ZS_TASKS)
        print("zero-shot:", res["zero_shot"], "mean", res["zero_shot_mean"], flush=True)

    os.makedirs(a.out, exist_ok=True)
    tag = a.tag or f"{a.model.split('/')[-1]}_{a.method}" + (f"_w{a.bits}g{a.group}" if a.method in ("rtn", "awq", "omniquant") else "") + (f"_w{a.bits}" if a.method == "quip" else "") \
        + (f"_{os.path.basename(a.compressed.rstrip('/'))}" if a.method == "seedlm" else "")
    json.dump(res, open(os.path.join(a.out, tag + ".json"), "w"), indent=1)
    print("saved", os.path.join(a.out, tag + ".json"))


if __name__ == "__main__":
    main()
