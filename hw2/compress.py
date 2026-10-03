"""Compress all decoder Linear weights of a HF causal LM with SeedLM."""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM

from seedlm import Config, compress, decompress, get_tables


def linear_layers(model):
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and "lm_head" not in n and "embed" not in n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, choices=[3, 4], default=4, help="paper configs, Table 1")
    ap.add_argument("--K", type=int, default=None, help="override LFSR length (faster, worse)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--reverse", action="store_true",
                    help="go through the layers from the last one; run a second process with "
                         "--reverse --device cuda:1 next to a normal one to use two GPUs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-layers", type=int, default=None, help="only first N Linear layers (debug)")
    ap.add_argument("--kernel", choices=["eager", "triton"], default="eager",
                    help="triton = fused search kernel (triton_search.py), much faster")
    ap.add_argument("--compile", action="store_true", help="torch.compile the eager scoring (not with --kernel triton)")
    ap.add_argument("--block-chunk", type=int, default=4096)
    ap.add_argument("--elems", type=int, default=1 << 25, help="max blocks*seeds per chunk")
    a = ap.parse_args()

    cfg = Config.paper(a.bits, a.K)
    os.makedirs(os.path.join(a.out, "layers"), exist_ok=True)
    json.dump(dict(model=a.model, C=cfg.C, P=cfg.P, K=cfg.K, bits=cfg.bits_per_weight),
              open(os.path.join(a.out, "meta.json"), "w"))
    assert torch.cuda.is_available(), "compress.py runs on a GPU only (no CPU fallback)"
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                 device_map={"": a.device})
    layers = linear_layers(model)[: a.max_layers]
    if a.reverse:
        layers = layers[::-1]
    print(f"{len(layers)} Linear layers, config {cfg}, {cfg.bits_per_weight:.2f} bits/weight", flush=True)

    log, t_start = [], time.time()
    tables = get_tables(cfg, a.device)
    if a.kernel == "triton":
        import triton_search
        rel = triton_search.check_against_eager(cfg, a.device)  # aborts if the kernel is wrong
        print(f"triton kernel self-check passed (total error differs by {rel:.2e} from eager)", flush=True)
    for tmp in os.listdir(os.path.join(a.out, "layers")):
        if tmp.endswith(".tmp"):
            os.remove(os.path.join(a.out, "layers", tmp))
    for name, mod in layers:
        path = os.path.join(a.out, "layers", name + ".pt")
        if os.path.exists(path):
            continue
        t0 = time.time()
        W = mod.weight.data.float()
        c = compress(W, cfg, tables, a.block_chunk, a.elems, a.compile, a.kernel)
        rel = ((decompress(c, a.device) - W).norm() / W.norm()).item()
        torch.save(dict(shape=c.shape, seeds=c.seeds.cpu(), exps=c.exps.cpu(), q=c.q.cpu()), path + ".tmp")
        os.replace(path + ".tmp", path)
        log.append(dict(layer=name, numel=W.numel(), rel_err=rel, sec=time.time() - t0))
        print(f"[{len(log)}/{len(layers)}] {name} {tuple(W.shape)} rel_err={rel:.4f} "
              f"{time.time() - t0:.1f}s", flush=True)
    json.dump(log, open(os.path.join(a.out, "compress_log_rev.json" if a.reverse else "compress_log.json"), "w"), indent=1)
    print(f"done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
