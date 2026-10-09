"""OmniQuant baseline (Shao et al., ICLR 2024), weight-only setting (W4A16 / W3A16), ported to Qwen3.

Ported from the official repo (OpenGVLab/OmniQuant: quantize/quantizer.py and quantize/omniquant.py).
For weight-only quantization the official scripts use only LWC (learnable weight clipping) without LET:
    python main.py ... --wbits 4 --abits 16 --lwc --epochs 20       (no --let, no --aug_loss)
The clipping strengths  sigmoid(upbound), sigmoid(lowbound)  (one pair per output channel, initialised
at 4.0) shrink the per-channel max and min before min-max quantization and are trained, layer by layer,
to minimise the MSE between the output of the full-precision layer (fed with the full-precision
activations) and the output of the layer with fake-quantized weights (fed with the activations of the
already quantized previous layers). AdamW, lr 1e-2, weight decay 0, batch size 1, 128 calibration
sequences of 2048 tokens from WikiText-2 train, 20 epochs.

Differences to the official code: bf16 autocast instead of fp16, the layer is called through
torch.func.functional_call instead of the QuantLinear wrappers, only Qwen3-style decoder layers.
"""
import copy
import gc
import random

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.func import functional_call

from awq import _Catcher, _run, _Stop

CLIPMIN = 1e-5
INIT = 4.0
LINEARS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def round_ste(x):
    return (x.round() - x).detach() + x


def lwc_quant(W, up, low, bits, group=0):
    """Fake quantization of W (out, in) with learnable clipping (UniformAffineQuantizer.fake_quant)."""
    out_f, in_f = W.shape
    g = in_f if group <= 0 else group
    x = W.reshape(-1, g)
    xmin, xmax = x.amin(1, keepdim=True), x.amax(1, keepdim=True)
    xmax, xmin = torch.sigmoid(up) * xmax, torch.sigmoid(low) * xmin
    scale = ((xmax - xmin) / (2 ** bits - 1)).clamp(CLIPMIN, 1e4)
    zero = (-xmin / scale).clamp(-1e4, 1e4).round()
    q = (round_ste(x / scale) + zero).clamp(0, 2 ** bits - 1)
    return ((q - zero) * scale).reshape(out_f, in_f)


def calib_wikitext(tok, n, seqlen, seed=2):
    """Random windows of the WikiText-2 train set, like get_wikitext2 in the official repo."""
    random.seed(seed)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    enc = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    ids = []
    for _ in range(n):
        i = random.randint(0, enc.shape[1] - seqlen - 1)
        ids.append(enc[0, i:i + seqlen])
    return torch.stack(ids)


def _get(layer, name):
    m = layer
    for p in name.split("."):
        m = getattr(m, p)
    return m


def omniquant(model, tok, bits: int, group: int = 0, n_samples: int = 128, seqlen: int = 2048,
              epochs: int = 20, lr: float = 1e-2, device: str = "cuda:0", log=print):
    """Quantize all decoder Linear layers of `model` (on `device`) in place with OmniQuant-LWC."""
    ids = calib_wikitext(tok, n_samples, seqlen).to(device)
    layers = model.model.layers
    store = {}
    layers[0] = _Catcher(layers[0], store)
    try:
        with torch.no_grad():
            model(input_ids=ids[:1], use_cache=False)
    except _Stop:
        pass
    layers[0] = layers[0].layer
    kwargs = store["kwargs"]
    with torch.no_grad():
        quant_inps = model.model.embed_tokens(ids)  # input of the quantized stack
        fp_inps = quant_inps.clone()  # input of the full-precision stack

    for li, layer in enumerate(layers):
        names = [n for n in LINEARS]
        with torch.no_grad():  # targets: full-precision layer on full-precision input
            for j in range(n_samples):
                fp_inps[j] = _run(layer, fp_inps[j:j + 1], kwargs)[0]
        layer32 = copy.deepcopy(layer).float().requires_grad_(False)
        params = {}
        for n in names:
            out_f, in_f = _get(layer, n).weight.shape
            rows = out_f * (1 if group <= 0 else in_f // group)
            params[n] = (torch.nn.Parameter(torch.full((rows, 1), INIT, device=device)),
                         torch.nn.Parameter(torch.full((rows, 1), INIT, device=device)))
        w32 = {n: _get(layer32, n).weight.detach() for n in names}
        opt = torch.optim.AdamW([p for pair in params.values() for p in pair], lr=lr, weight_decay=0)
        power = fp_inps.float().pow(2).mean().item()  # mean square of the targets (scale of the loss)
        peak = fp_inps.float().abs().max().item()  # largest |activation| (massive activations show up here)
        for ep in range(epochs):
            losses, losses_no0 = [], []
            for j in range(n_samples):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    wq = {f"{n}.weight": lwc_quant(w32[n], *params[n], bits, group) for n in names}
                    out = functional_call(layer32, wq, args=(quant_inps[j:j + 1],), kwargs=kwargs)
                    out = out[0] if isinstance(out, tuple) else out
                    loss = F.mse_loss(out.float(), fp_inps[j:j + 1].float())
                if not torch.isfinite(loss):
                    raise RuntimeError(f"OmniQuant: loss is not finite at layer {li}, epoch {ep}")
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.detach())
                losses_no0.append(F.mse_loss(out[:, 1:].float(), fp_inps[j:j + 1, 1:].float()).detach())
            mean = torch.stack(losses).mean().item()
            log(f"omniquant layer {li + 1}/{len(layers)} epoch {ep + 1}/{epochs} loss {mean:.6f} "
                f"| without 1st token {torch.stack(losses_no0).mean().item():.6f} "
                f"| relative {mean / power:.2e} | target power {power:.3g}, max|x| {peak:.3g}")
        with torch.no_grad():  # quantize for real, then propagate the quantized stack
            for n in names:
                m = _get(layer, n)
                m.weight.data = lwc_quant(m.weight.data.float(), *params[n], bits, group).to(m.weight.dtype)
            for j in range(n_samples):
                quant_inps[j] = _run(layer, quant_inps[j:j + 1], kwargs)[0]
        del layer32, w32, params, opt
        gc.collect()
        torch.cuda.empty_cache()
    return model
