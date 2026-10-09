"""QuIP# baseline (Tseng et al., ICML 2024) without fine-tuning, via the official repository.

The official pipeline is written for Llama and pins old torch/transformers, so only its per-matrix
quantizer (lib/algo/quip.py: randomized Hadamard incoherence processing + LDLQ rounding onto the E8P
lattice codebook; 4 bit = E8P12RVQ4B, 3 bit = E8P12RVQ3B) is used. Hessians H = E[x x^T] of the input of
every linear layer are collected here, layer by layer, and the matrices are quantized like the official
`quantize_finetune_llama.py` does with its fusion of q/k/v and gate/up and normalisation of every
matrix by its RMS. The decoded weights are written back, so this measures accuracy (fake quantization).

Setup (once):
    git clone https://github.com/Cornell-RelaxML/quip-sharp third_party/quip-sharp
    pip install primefac scipy        # primefac: only needed for sizes without a Hadamard matrix (Qwen3-14B MLP)
The CUDA extensions of the repo (quiptools_cuda, fast_hadamard_transform) are NOT needed here: they are
used only for fast inference, and small pure-torch stand-ins are installed for the imports.

Differences to the paper's QuIP# setup: no fine-tuning (as in the SeedLM paper), Hessians from
`n_samples` x 2048 tokens of the Pile instead of 6144 x 4096 tokens of RedPajama, and the incoherence
processing falls back from the Hadamard to the Kronecker/butterfly variant for sizes without a Hadamard
matrix (e.g. 17408 = 17 x 1024). The repo is GPL-3.0 and is used as an external dependency.
"""
import gc
import time
import importlib
import os
import sys
import types

import torch

from awq import _Catcher, _run, _Stop, calib_ids

HAD_K = [172, 156, 140, 124, 116, 108, 60, 52, 36, 28, 20, 12]  # same order as get_hadK in the repo


def had_supported(n: int) -> bool:
    pow2 = lambda v: v > 0 and v & (v - 1) == 0  # noqa: E731
    for k in HAD_K:
        if n % k == 0:
            return pow2(n // k)
    return pow2(n)


def _install_shims():
    """Stand-ins for modules that the repo imports but that are not needed for quantization."""
    if not hasattr(torch.library, "impl_abstract"):
        torch.library.impl_abstract = torch.library.register_fake
    for name in ("quiptools_cuda", "glog", "fast_hadamard_transform"):
        try:
            importlib.import_module(name)
        except ImportError:
            mod = types.ModuleType(name)
            if name == "glog":
                for fn in ("info", "warning", "error", "debug", "set_verbosity"):
                    setattr(mod, fn, lambda *a, **k: None)
            if name == "fast_hadamard_transform":
                def hadamard_transform(x, scale=1.0):  # Walsh-Hadamard along the last (power of 2) dim
                    n = x.shape[-1]
                    y = x.reshape(-1, n)
                    h = 1
                    while h < n:
                        y = y.reshape(-1, n // (2 * h), 2, h)
                        y = torch.stack((y[:, :, 0] + y[:, :, 1], y[:, :, 0] - y[:, :, 1]), 2)
                        h *= 2
                    return (y.reshape(x.shape) * scale).to(x.dtype)
                mod.hadamard_transform = hadamard_transform
            sys.modules[name] = mod
    try:
        import scipy.stats  # noqa: F401  (matmul_kron.py does `import scipy` and then uses scipy.stats)
    except ImportError:
        pass
    try:
        importlib.import_module("primefac")
    except ImportError:
        stub = types.ModuleType("primefac")

        def _missing(*a, **k):
            raise ImportError("this model needs the Kronecker incoherence processing of QuIP#: pip install primefac")
        stub.primefac = _missing
        sys.modules["primefac"] = stub


def load_quip(repo: str):
    """Import lib.algo.quip and lib.codebook from the cloned repo without running lib/__init__ files
    that pull in lm_eval, transformers 4.40 and the CUDA extensions."""
    if "lib.algo.quip" in sys.modules:
        return sys.modules["lib.algo.quip"], sys.modules["lib.codebook"]
    _install_shims()
    repo = os.path.abspath(repo)
    assert os.path.isdir(os.path.join(repo, "lib")), f"{repo} is not a clone of Cornell-RelaxML/quip-sharp"
    sys.path.insert(0, repo)
    lib = types.ModuleType("lib")
    lib.__path__ = [os.path.join(repo, "lib")]
    sys.modules["lib"] = lib
    utils = types.ModuleType("lib.utils")
    utils.__path__ = [os.path.join(repo, "lib", "utils")]
    sys.modules["lib.utils"] = utils
    lib.utils = utils
    for name in ("math_utils", "misc", "matmul_kron", "matmul_had"):
        mod = importlib.import_module(f"lib.utils.{name}")
        if not hasattr(mod, "torch"):  # matmul_kron.py of the repo uses torch without importing it
            mod.torch = torch
        utils.__dict__.update({k: v for k, v in vars(mod).items() if not k.startswith("__")})
    codebook = importlib.import_module("lib.codebook")
    quip = importlib.import_module("lib.algo.quip")
    return quip, codebook


def quip_args(incoh_mode="had", sigma_reg=1e-2, tune_iters=10):
    """Defaults of quantize_finetune_llama.py (fine-tuning switched off, no low-rank correction)."""
    return types.SimpleNamespace(
        incoh_mode=incoh_mode, sigma_reg=sigma_reg, lora_rank=0, scale_override=-1, resid_scale_override=-1,
        quip_tune_iters=tune_iters, use_fp64=False, full_svd=False, no_use_buffered=False, rescale_WH=False,
        lowmem_ldlq=False, save_pfx="/tmp")


@torch.no_grad()
def quip_matrix(quip, utils, cb, weights, H, sigma_reg, tune_iters, device):
    """Quantize stacked weights that share the input Hessian H; returns the decoded weights."""
    scales = [w.float().square().mean().sqrt() for w in weights]
    W = torch.vstack([w.float() / s for w, s in zip(weights, scales)])
    H = utils.regularize_H(H.clone(), H.shape[0], sigma_reg)
    mode = "had" if had_supported(W.shape[0]) and had_supported(W.shape[1]) else "kron"
    hatW, _ = quip.quantize(H, W, 0, cb, quip_args(mode, sigma_reg, tune_iters), device)
    out, cur = [], 0
    for w, s in zip(weights, scales):
        out.append((hatW[cur:cur + w.shape[0]].to(device) * s).to(w.dtype))
        cur += w.shape[0]
    return out


@torch.no_grad()
def quip_quantize(model, tok, bits: int, repo: str, n_samples: int = 128, seqlen: int = 2048,
                  source: str = "pile", sigma_reg: float = 1e-2, tune_iters: int = 10,
                  device: str = "cuda:0", log=print, ckpt_dir: str = None):
    """ckpt_dir: every quantized layer is saved there, so an interrupted run resumes at the first missing layer."""
    quip, codebook = load_quip(repo)
    utils = sys.modules["lib.utils"]
    cb = codebook.get_codebook({4: "E8P12RVQ4B", 3: "E8P12RVQ3B"}[bits])
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
    bs = 4

    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    t_all = time.time()
    for li, layer in enumerate(layers):
        att, mlp = layer.self_attn, layer.mlp
        mats = [att.q_proj, att.k_proj, att.v_proj, att.o_proj, mlp.gate_proj, mlp.up_proj, mlp.down_proj]
        ckpt = os.path.join(ckpt_dir, f"layer_{li}.pt") if ckpt_dir else None
        if ckpt and os.path.exists(ckpt):  # done before: only propagate the full-precision activations
            outs = torch.cat([_run(layer, inps[i:i + bs], kwargs) for i in range(0, n_samples, bs)])
            for m, w in zip(mats, torch.load(ckpt, map_location=device)):
                m.weight.data = w.to(m.weight.dtype)
            inps = outs
            del outs
            log(f"quip# layer {li + 1}/{len(layers)}: loaded from {ckpt}")
            continue
        t_layer = time.time()
        taps = {"qkv": att.q_proj, "o": att.o_proj, "up": mlp.gate_proj, "down": mlp.down_proj}
        H, count, hooks = {}, {"n": 0}, []
        for k, m in taps.items():
            H[k] = torch.zeros(m.in_features, m.in_features, dtype=torch.float32, device=device)

            def hook(mod, inp, k=k):
                x = inp[0].reshape(-1, inp[0].shape[-1])
                for i in range(0, x.shape[0], 8192):
                    xc = x[i:i + 8192].float()
                    H[k].addmm_(xc.T, xc)
                if k == "qkv":
                    count["n"] += x.shape[0]
            hooks.append(m.register_forward_pre_hook(hook))
        outs = torch.cat([_run(layer, inps[i:i + bs], kwargs) for i in range(0, n_samples, bs)])
        for h in hooks:
            h.remove()
        for k in H:
            H[k] /= count["n"]

        groups = [("qkv", [att.q_proj, att.k_proj, att.v_proj]), ("o", [att.o_proj]),
                  ("up", [mlp.up_proj, mlp.gate_proj]), ("down", [mlp.down_proj])]
        log(f"quip# layer {li + 1}/{len(layers)}: Hessians collected from {count['n']} tokens")
        for key, mods in groups:
            t0 = time.time()
            new = quip_matrix(quip, utils, cb, [m.weight.data for m in mods], H[key], sigma_reg, tune_iters, device)
            log(f"quip# layer {li + 1}/{len(layers)}: {key} {tuple(torch.vstack([m.weight.data for m in mods]).shape)} "
                f"done in {time.time() - t0:.0f}s")
            for m, w in zip(mods, new):
                m.weight.data = w
        if ckpt:
            torch.save([m.weight.data.cpu() for m in mats], ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
        inps = outs
        del H, outs
        gc.collect()
        torch.cuda.empty_cache()
        log(f"quip# layer {li + 1}/{len(layers)} finished in {time.time() - t_layer:.0f}s "
            f"(total {(time.time() - t_all) / 60:.1f} min)")
    return model
