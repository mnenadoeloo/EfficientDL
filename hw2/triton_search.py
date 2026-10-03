import importlib.util
import os

import torch

try:
    import triton  # noqa: F401
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

BB, BS, WARPS = 32, 32, 4
_GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_gen_kernels")
_kernels = {}


def _source(C: int, P: int) -> str:
    pairs = [(p, q) for p in range(P) for q in range(p, P)]
    L = ["import triton", "import triton.language as tl", "", "", "@triton.jit",
         "def kern(w_ptr, wn2_ptr, pinv_ptr, g_ptr, out_ptr, nb, N, BB: tl.constexpr, BS: tl.constexpr):",
         "    pid = tl.program_id(0)",
         "    b = pid * BB + tl.arange(0, BB)",
         "    bm = b < nb",
         "    b64 = b.to(tl.int64)",
         "    wn2 = tl.load(wn2_ptr + b, mask=bm, other=0.0)"]
    L += [f"    w{c} = tl.load(w_ptr + b64 * {C} + {c}, mask=bm, other=0.0)[:, None]" for c in range(C)]
    L += ["    best_e = tl.full([BB], float('inf'), tl.float32)",
          "    best_j = tl.zeros([BB], tl.int32)",
          "    for s0 in range(0, N, BS):",
          "        s = s0 + tl.arange(0, BS)",
          "        sm = s < N"]
    for p in range(P):
        for c in range(C):
            L.append(f"        a = tl.load(pinv_ptr + {p * C + c} * N + s, mask=sm, other=0.0)[None, :]")
            L.append(f"        t{p} = w{c} * a" if c == 0 else f"        t{p} += w{c} * a")
    amax = "tl.abs(t0)"
    for p in range(1, P):
        amax = f"tl.maximum({amax}, tl.abs(t{p}))"
    L += [f"        amax = {amax}",
          "        ef = (amax.to(tl.int32, bitcast=True) >> 23) & 255",  
          "        e = tl.minimum(tl.maximum(ef - 129, -8), 7)", 
          "        scale = ((e + 127) << 23).to(tl.float32, bitcast=True)", 
          "        inv = ((127 - e) << 23).to(tl.float32, bitcast=True)"] 
    for p in range(P):
        L.append(f"        q{p} = tl.minimum(tl.maximum(tl.floor(t{p} * inv + 0.5), -8.0), 7.0)")
        L.append(f"        d{p} = q{p} * scale - t{p}")
    for k in range(len(pairs)):
        L.append(f"        g{k} = tl.load(g_ptr + {k} * N + s, mask=sm, other=0.0)[None, :]")
    qt = " + ".join(f"g{k} * t{p} * t{q}" for k, (p, q) in enumerate(pairs))
    qd = " + ".join(f"g{k} * d{p} * d{q}" for k, (p, q) in enumerate(pairs))
    L += [f"        err = wn2[:, None] - ({qt}) + ({qd})",
          "        err = tl.where(sm[None, :], err, float('inf'))",
          "        m = tl.min(err, axis=1)",
          "        am = tl.argmin(err, axis=1).to(tl.int32)",
          "        upd = m < best_e",
          "        best_e = tl.where(upd, m, best_e)",
          "        best_j = tl.where(upd, s0 + am, best_j)",
          "    tl.store(out_ptr + b, best_j, mask=bm)", ""]
    return "\n".join(L)


def _get_kernel(C: int, P: int):
    if (C, P) not in _kernels:
        os.makedirs(_GEN_DIR, exist_ok=True)
        path = os.path.join(_GEN_DIR, f"k_C{C}_P{P}.py")
        tmp = f"{path}.{os.getpid()}.tmp"
        open(tmp, "w").write(_source(C, P))
        os.replace(tmp, path)
        spec = importlib.util.spec_from_file_location(f"k_C{C}_P{P}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _kernels[(C, P)] = mod.kern
    return _kernels[(C, P)]


def _prepare(tb):
    """Kernel-friendly layouts, cached on the Tables object: pinv as (P*C, N), G as (P(P+1)/2, N)
    with off-diagonal entries doubled (so that x^T G x is a plain sum of g_k * x_p * x_q)."""
    if not hasattr(tb, "_tri"):
        P, C = tb.cfg.P, tb.cfg.C
        pinv = tb.pinv.permute(1, 2, 0).reshape(P * C, tb.N).contiguous()
        g = torch.stack([tb.Gd[p, q] * (1.0 if p == q else 2.0) for p in range(P) for q in range(p, P)]).contiguous()
        tb._tri = (pinv, g)
    return tb._tri


@torch.no_grad()
def search(blocks: torch.Tensor, tb, bb: int = None, bs: int = None, warps: int = None) -> torch.Tensor:
    """blocks: (nb, C) fp32 cuda tensor -> index (in the LFSR cycle) of the best seed per block."""
    assert HAS_TRITON, "triton is not installed"
    nb, C = blocks.shape
    P, N = tb.cfg.P, tb.N
    pinv, g = _prepare(tb)
    blocks = blocks.contiguous()
    wn2 = (blocks * blocks).sum(-1).contiguous()
    out = torch.empty(nb, dtype=torch.int32, device=blocks.device)
    bb, bs, warps = bb or BB, bs or BS, warps or WARPS
    _get_kernel(C, P)[(triton.cdiv(nb, bb),)](blocks, wn2, pinv, g, out, nb, N, BB=bb, BS=bs, num_warps=warps)
    return out.long()


def check_against_eager(cfg, device: str, nb: int = 2048, seed: int = 0, tol: float = 1e-3) -> float:
    """Run both searches on random Gaussian blocks. Returns the relative difference of the total
    reconstruction error (must be tiny) and raises if it is above tol."""
    import seedlm
    tb = seedlm.get_tables(cfg, device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    W = (torch.randn(nb, cfg.C, generator=g) * 0.03).to(device)
    errs = []
    for kernel in ("eager", "triton"):
        c = seedlm.compress(W, cfg, tb, kernel=kernel)
        errs.append(((seedlm.decompress(c, device) - W) ** 2).sum().item())
    rel = abs(errs[1] - errs[0]) / errs[0]
    if rel > tol:
        raise RuntimeError(f"triton kernel disagrees with the eager search: err {errs[1]:.6g} vs {errs[0]:.6g}")
    return rel


def tune_tiles(cfg, device: str, nb: int = 1 << 16):
    """Time a few tile shapes and print them (run once, then edit BB/BS/WARPS above)."""
    import time

    import seedlm
    tb = seedlm.get_tables(cfg, device)
    blocks = (torch.randn(nb, cfg.C) * 0.03).to(device)
    for bb, bs, w in [(16, 32, 2), (32, 32, 4), (32, 64, 4), (64, 32, 4), (64, 64, 8), (128, 32, 8)]:
        try:
            search(blocks, tb, bb, bs, w)
            torch.cuda.synchronize()
            t0 = time.time()
            search(blocks, tb, bb, bs, w)
            torch.cuda.synchronize()
            dt = time.time() - t0
            print(f"BB={bb:3d} BS={bs:3d} warps={w}: {dt:.3f}s  {nb * tb.N / dt / 1e9:.1f} G pairs/s", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"BB={bb} BS={bs} warps={w}: failed ({type(ex).__name__})", flush=True)


if __name__ == "__main__":
    import seedlm
    cfg = seedlm.Config.paper(4)
    print("rel diff vs eager:", check_against_eager(cfg, "cuda:0"))
    tune_tiles(cfg, "cuda:0")
