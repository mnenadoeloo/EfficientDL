"""AWQ baseline (Lin et al., MLSys 2024), re-implemented as fake quantization for HF decoder models."""
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

N_GRID = 20

def pseudo_quantize(w: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """Asymmetric min-max RTN over rows of `group` elements (group <= 0: the whole input channel)."""
    shape = w.shape
    g = shape[-1] if group <= 0 else group
    x = w.float().reshape(-1, g)
    lo, hi = x.amin(1, keepdim=True), x.amax(1, keepdim=True)
    scale = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-5)
    zp = torch.round(-lo / scale)
    q = (torch.round(x / scale) + zp).clamp(0, 2 ** bits - 1)
    return ((q - zp) * scale).reshape(shape).to(w.dtype)


def calib_ids(tok, n: int, seqlen: int, source: str) -> torch.Tensor:
    """n x seqlen token ids: short documents are concatenated and cut into blocks, like llm-awq."""
    if source == "pile":
        try:
            ds = load_dataset("mit-han-lab/pile-val-backup", split="validation").shuffle(seed=42)
        except Exception as ex:  # noqa: BLE001
            raise RuntimeError("could not load the Pile calibration set; the file is zstd-compressed, "
                               "so `pip install zstandard` (or use --calib wikitext)") from ex
        texts = (r["text"] for r in ds)
    else:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        texts = (t for t in ds["text"] if t.strip())
    parts, total = [], 0
    for t in texts:
        ids = tok(t.strip(), add_special_tokens=False).input_ids
        if not ids or len(ids) > seqlen:
            continue
        parts.append(torch.tensor(ids))
        total += len(ids)
        if total >= n * seqlen:
            break
    assert total >= n * seqlen, f"calibration set too small: {total} tokens"
    return torch.cat(parts)[: n * seqlen].reshape(n, seqlen)


class _Stop(Exception):
    pass


class _Catcher(nn.Module):
    """Replaces decoder layer 0 once, to record the keyword arguments the model passes to a layer."""

    def __init__(self, layer, store):
        super().__init__()
        self.layer, self.store = layer, store

    def forward(self, hidden_states, *args, **kwargs):
        self.store["kwargs"] = kwargs
        raise _Stop

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)


def _run(layer, x, kwargs):
    out = layer(x, **kwargs)
    return out[0] if isinstance(out, tuple) else out


def _chunks(fn, x, size=8192):
    return torch.cat([fn(x[i:i + size]) for i in range(0, x.shape[0], size)])


@torch.no_grad()
def _search_scale(x, linears, fn, bits, group):
    """Return the scale vector (one entry per input channel) with the smallest output error."""
    x_mean = x.abs().float().mean(0)
    org = _chunks(fn, x).float()
    saved = [l.weight.data.clone() for l in linears]
    best_loss, best_s = float("inf"), None
    for r in range(N_GRID):
        s = x_mean.pow(r / N_GRID).clamp(min=1e-4)
        s = s / (s.max() * s.min()).sqrt()
        for l, w0 in zip(linears, saved):
            w = pseudo_quantize(w0.float() * s[None, :], bits, group) / s[None, :]
            l.weight.data = w.to(w0.dtype)
        loss = (org - _chunks(fn, x).float()).pow(2).mean().item()
        if loss < best_loss:
            best_loss, best_s = loss, s.clone()
    for l, w0 in zip(linears, saved):
        l.weight.data = w0
    return best_s


@torch.no_grad()
def _auto_clip(w, x, bits, group, max_shrink=0.5, n_tok=512):
    """Per output channel (and quantization group) clipping threshold; clips `w` in place."""
    out_f, in_f = w.shape
    g = in_f if group <= 0 else group
    x = x[:: max(1, x.shape[0] // n_tok)][:n_tok].float().reshape(1, -1, in_f // g, g)
    wg = w.reshape(out_f, 1, in_f // g, g)
    batch = max(1, min(256, (1 << 27) // (x.shape[1] * in_f)))
    best = []
    for i in range(0, out_f, batch):
        wb = wg[i:i + batch].float()
        org_max = wb.abs().amax(-1, keepdim=True)
        org_out = (x * wb).sum(-1)
        best_max, best_err = org_max.clone(), torch.full_like(org_max, float("inf"))
        for k in range(int(max_shrink * N_GRID)):
            m = org_max * (1 - k / N_GRID)
            qw = pseudo_quantize(torch.clamp(wb, -m, m).reshape(-1, g), bits, 0).reshape(wb.shape)
            err = (((x * qw).sum(-1) - org_out) ** 2).mean(1).reshape(best_err.shape)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_max = torch.where(better, m, best_max)
        best.append(best_max)
    m = torch.cat(best).reshape(out_f, in_f // g, 1)
    w.data = torch.clamp(w.reshape(out_f, in_f // g, g), -m.to(w.dtype), m.to(w.dtype)).reshape(out_f, in_f)


@torch.no_grad()
def awq_quantize(model, tok, bits: int, group: int = 0, n_samples: int = 128, seqlen: int = 512,
                 source: str = "pile", max_tokens: int = 32768, device: str = "cuda:0", seed: int = 0):
    """Quantize all decoder Linear layers of `model` (already on `device`) in place, AWQ-style."""
    torch.manual_seed(seed)
    ids = calib_ids(tok, n_samples, seqlen, source).to(device)
    layers = model.model.layers
    store = {}
    layers[0] = _Catcher(layers[0], store)
    try:
        model(input_ids=ids[:1], use_cache=False)
    except _Stop:
        pass
    layers[0] = layers[0].layer
    kwargs = store["kwargs"]
    inps = model.model.embed_tokens(ids)
    bs = 16
    keep = max(1, max_tokens * bs // n_samples)

    for li, layer in enumerate(layers):
        att, mlp = layer.self_attn, layer.mlp
        names = {"qkv": att.q_proj, "o": att.o_proj, "gate": mlp.gate_proj, "down": mlp.down_proj}
        feats, hooks = {k: [] for k in names}, []
        for k, m in names.items():
            def hook(mod, inp, k=k):
                x = inp[0].reshape(-1, inp[0].shape[-1])
                feats[k].append(x[torch.randperm(x.shape[0], device=x.device)[:keep]].clone())
            hooks.append(m.register_forward_pre_hook(hook))
        outs = torch.cat([_run(layer, inps[i:i + bs], kwargs) for i in range(0, n_samples, bs)])
        for h in hooks:
            h.remove()
        feats = {k: torch.cat(v) for k, v in feats.items()}

        q, k_, v = att.q_proj, att.k_proj, att.v_proj
        groups = [
            ("qkv", [q, k_, v], lambda x: torch.cat([F.linear(x, m.weight) for m in (q, k_, v)], -1),
             layer.input_layernorm, None),
            ("gate", [mlp.gate_proj, mlp.up_proj], lambda x: mlp(x), layer.post_attention_layernorm, None),
            ("down", [mlp.down_proj], lambda x: F.linear(x, mlp.down_proj.weight), None, mlp.up_proj),
        ]
        scales = [_search_scale(feats[name], lins, fn, bits, group) for name, lins, fn, _, _ in groups]
        for (name, lins, _, norm, prev_lin), s in zip(groups, scales):  # fold the scales
            dt = lins[0].weight.dtype
            if norm is not None:
                norm.weight.data = (norm.weight.data.float() / s).to(norm.weight.dtype)
            else:
                prev_lin.weight.data = (prev_lin.weight.data.float() / s[:, None]).to(prev_lin.weight.dtype)
            for l in lins:
                l.weight.data = (l.weight.data.float() * s[None, :]).to(dt)
            feats[name] = (feats[name].float() / s).to(dt)

        for name, m in [("qkv", att.v_proj), ("o", att.o_proj), ("gate", mlp.gate_proj),
                        ("gate", mlp.up_proj), ("down", mlp.down_proj)]:
            _auto_clip(m.weight, feats[name], bits, group)
        for m in (att.q_proj, att.k_proj, att.v_proj, att.o_proj, mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            m.weight.data = pseudo_quantize(m.weight.data, bits, group)
        inps = outs
        del feats, outs
        gc.collect()
        torch.cuda.empty_cache()
        print(f"awq layer {li + 1}/{len(layers)}", flush=True)
    return model
