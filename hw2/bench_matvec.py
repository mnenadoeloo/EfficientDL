"""GPU analogue of Table 5 of the paper: y = W x (batch 1, the memory-bound step of LLM decoding) with
weights stored as BF16, as plain 4-bit RTN, or as 4-bit SeedLM that is decoded inside the kernel.

SeedLM word (32 bit per block of C = 8 weights = 4 bit/weight, config C=8, P=3, K=16):
    bits  0-15 : j, the index of the seed in the LFSR cycle
    bits 16-19 : shared exponent e + 8
    bits 20-31 : three 4-bit two's-complement coefficients
The kernel gathers the 24 LFSR values seq[j+1 .. j+24] from a 256 KB table (it stays in L1/L2),
builds U (8 x 3) and computes w = U @ (q * 2^e) in registers, so only 4 bit/weight cross HBM.

    python bench_matvec.py            # correctness check + timing table -> results/matvec.csv

The FPGA of the paper has idle MACs; an A100 has far less spare compute per byte, so the result of
this benchmark need not be 4x. Nothing here is a claim about the FPGA numbers (Tables 4, 5).
"""
import argparse
import csv
import itertools
import os

import torch
import triton
import triton.language as tl

import seedlm
from seedlm import Config, compress, decompress, get_tables

C, P, K = 8, 3, 16
_GEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_gen_kernels")


# --------------------------------------------------------------------------- packing
def pack_seedlm(c, tb) -> torch.Tensor:
    """Compressed (C=8, P=3, K=16) -> int32 words of shape (rows, cols // 8)."""
    rows, cols = c.shape
    assert (c.cfg.C, c.cfg.P, c.cfg.K) == (C, P, K) and cols % C == 0
    dev = tb.pos.device
    j = tb.pos[(c.seeds.to(torch.int32) & 0xFFFF).long().to(dev)]  # state -> index in the LFSR cycle
    e = c.exps.to(dev).long() + 8
    q = c.q.to(dev).long() & 15
    word = j | (e << 16) | (q[:, 0] << 20) | (q[:, 1] << 24) | (q[:, 2] << 28)  # int64, < 2**32
    word = (word + 2 ** 31) % 2 ** 32 - 2 ** 31  # wrap to the int32 range (bit 31 = sign)
    return word.to(torch.int32).reshape(rows, cols // C)


def pack_rtn4(W: torch.Tensor):
    """Per-row asymmetric 4-bit RTN -> (int32 words (rows, cols//8), scale (rows), zero (rows))."""
    rows, cols = W.shape
    x = W.float()
    lo, hi = x.amin(1), x.amax(1)
    scale = ((hi - lo) / 15).clamp_min(1e-8)
    zp = torch.round(-lo / scale)
    q = (torch.round(x / scale[:, None]) + zp[:, None]).clamp(0, 15).to(torch.int64).reshape(rows, cols // 8, 8)
    word = torch.zeros(rows, cols // 8, dtype=torch.int64, device=W.device)
    for i in range(8):
        word |= q[:, :, i] << (4 * i)
    word = (word + 2 ** 31) % 2 ** 32 - 2 ** 31
    return word.to(torch.int32), scale, zp


# --------------------------------------------------------------------------- kernels
def _seed_kernel_source() -> str:
    L = ["import triton", "import triton.language as tl", "", "", "@triton.jit",
         "def seed_matvec(w_ptr, tab_ptr, x_ptr, y_ptr, R, NCB, N, BR: tl.constexpr, CB: tl.constexpr):",
         "    rows = tl.program_id(0) * BR + tl.arange(0, BR)",
         "    rm = rows < R",
         "    acc = tl.zeros([BR], tl.float32)",
         "    for cb0 in range(0, NCB, CB):",
         "        cbs = cb0 + tl.arange(0, CB)",
         "        cm = cbs < NCB",
         "        m2 = rm[:, None] & cm[None, :]",
         "        word = tl.load(w_ptr + rows[:, None].to(tl.int64) * NCB + cbs[None, :], mask=m2, other=0)",
         "        j = word & 65535",
         "        e = ((word >> 16) & 15) - 8",
         "        scale = ((e + 127) << 23).to(tl.float32, bitcast=True)"]
    for p in range(P):
        L.append(f"        t{p} = ((((word >> {20 + 4 * p}) & 15) ^ 8) - 8).to(tl.float32) * scale")
    for k in range(C * P):
        L.append(f"        i{k} = j + {k + 1}")
        L.append(f"        i{k} = tl.where(i{k} >= N, i{k} - N, i{k})")
        L.append(f"        u{k} = (tl.load(tab_ptr + i{k}, mask=m2, other=32768).to(tl.float32) - 32768.0) * {1.0 / 32767.0}")
    for c in range(C):
        L.append(f"        xc = tl.load(x_ptr + cbs * {C} + {c}, mask=cm, other=0.0).to(tl.float32)")
        terms = " + ".join(f"u{c * P + p} * t{p}" for p in range(P))
        L.append(f"        acc += tl.sum(({terms}) * xc[None, :], axis=1)")
    L += ["    tl.store(y_ptr + rows, acc, mask=rm)", ""]
    return "\n".join(L)


def _load_seed_kernel():
    os.makedirs(_GEN, exist_ok=True)
    path = os.path.join(_GEN, "seed_matvec.py")
    tmp = f"{path}.{os.getpid()}.tmp"
    open(tmp, "w").write(_seed_kernel_source())
    os.replace(tmp, path)
    import importlib.util
    spec = importlib.util.spec_from_file_location("seed_matvec_gen", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.seed_matvec


@triton.jit
def _rtn4_matvec(w_ptr, sc_ptr, zp_ptr, x_ptr, y_ptr, R, NCB, BR: tl.constexpr, CB: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = rows < R
    sc = tl.load(sc_ptr + rows, mask=rm, other=0.0)
    zp = tl.load(zp_ptr + rows, mask=rm, other=0.0)
    acc = tl.zeros([BR], tl.float32)
    for cb0 in range(0, NCB, CB):
        cbs = cb0 + tl.arange(0, CB)
        cm = cbs < NCB
        m2 = rm[:, None] & cm[None, :]
        word = tl.load(w_ptr + rows[:, None].to(tl.int64) * NCB + cbs[None, :], mask=m2, other=0)
        for c in tl.static_range(8):
            nib = ((word >> (4 * c)) & 15).to(tl.float32)
            xc = tl.load(x_ptr + cbs * 8 + c, mask=cm, other=0.0).to(tl.float32)
            acc += tl.sum((nib - zp[:, None]) * sc[:, None] * xc[None, :], axis=1)
    tl.store(y_ptr + rows, acc, mask=rm)


def run_seed(kernel, words, tab, x, cfg):
    R, NCB = words.shape
    y = torch.empty(R, dtype=torch.float32, device=x.device)
    br, cb, nw = cfg
    kernel[(triton.cdiv(R, br),)](words, tab, x, y, R, NCB, tab.numel(), BR=br, CB=cb, num_warps=nw)
    return y


def run_rtn(words, sc, zp, x, cfg):
    R, NCB = words.shape
    y = torch.empty(R, dtype=torch.float32, device=x.device)
    br, cb, nw = cfg
    _rtn4_matvec[(triton.cdiv(R, br),)](words, sc, zp, x, y, R, NCB, BR=br, CB=cb, num_warps=nw)
    return y


# --------------------------------------------------------------------------- timing
def time_us(fn, iters=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    return sorted(ts)[len(ts) // 2]


def best_cfg(run, space=((4, 16, 2), (8, 16, 4), (8, 32, 4), (16, 32, 4), (16, 64, 8), (32, 32, 8), (2, 32, 2))):
    best = (float("inf"), None)
    for cfg in space:
        try:
            t = time_us(lambda: run(cfg), iters=15, warm=3)
        except Exception:  # noqa: BLE001  (a config may not compile or run out of registers)
            continue
        best = min(best, (t, cfg), key=lambda z: z[0])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/matvec.csv")
    ap.add_argument("--sizes", default="512x512,1024x1024,2048x2048,4096x4096,8192x8192,17408x5120,4096x14336")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "GPU only"
    dev = a.device
    torch.manual_seed(0)
    cfg = Config(C, P, K)
    tb = get_tables(cfg, dev)
    tab = tb.seq.to(torch.int32).contiguous()
    kernel = _load_seed_kernel()

    # 1) correctness on a matrix that really goes through compress()
    W = torch.randn(1024, 2048, device=dev) * 0.02
    c = compress(W, cfg, tb, kernel="triton")
    words = pack_seedlm(c, tb)
    x = torch.randn(2048, device=dev).to(torch.bfloat16)
    ref = decompress(c, dev) @ x.float()
    got = run_seed(kernel, words, tab, x, (8, 32, 4))
    rel = ((got - ref).norm() / ref.norm()).item()
    print(f"SeedLM kernel vs dense decompress: relative error {rel:.2e}")
    assert rel < 1e-3, "SeedLM matvec kernel is wrong"
    wr, sc, zp = pack_rtn4(W)
    deq = ((((wr.view(torch.int32)[:, :, None] >> (4 * torch.arange(8, device=dev))) & 15).float() - zp[:, None, None])
           * sc[:, None, None]).reshape(W.shape)
    ref2 = deq @ x.float()
    rel2 = ((run_rtn(wr, sc, zp, x, (8, 32, 4)) - ref2).norm() / ref2.norm()).item()
    print(f"RTN-4bit kernel vs dense dequantize: relative error {rel2:.2e}")
    assert rel2 < 1e-3, "RTN matvec kernel is wrong"

    # 2) timing (random packed weights: speed does not depend on the values)
    rows_out = []
    print(f"{'matrix':>12} {'BF16 us':>9} {'int4 us':>9} {'SeedLM us':>10} {'int4/BF16':>10} {'SeedLM/BF16':>12}")
    for size in a.sizes.split(","):
        R, N = map(int, size.split("x"))
        assert N % 8 == 0
        Wb = (torch.randn(R, N, device=dev) * 0.02).to(torch.bfloat16)
        x = torch.randn(N, device=dev).to(torch.bfloat16)
        t_bf = time_us(lambda: torch.mv(Wb, x))
        wr, sc, zp = pack_rtn4(Wb.float())
        words = torch.randint(-2 ** 31, 2 ** 31 - 1, (R, N // 8), dtype=torch.int32, device=dev)
        words = (words & ~65535) | torch.randint(0, tb.N, (R, N // 8), dtype=torch.int32, device=dev)
        t_rtn, c_rtn = best_cfg(lambda cf: run_rtn(wr, sc, zp, x, cf))
        t_sd, c_sd = best_cfg(lambda cf: run_seed(kernel, words, tab, x, cf))
        print(f"{size:>12} {t_bf:9.1f} {t_rtn:9.1f} {t_sd:10.1f} {t_bf / t_rtn:9.2f}x {t_bf / t_sd:11.2f}x", flush=True)
        rows_out.append(dict(matrix=size, bf16_us=t_bf, int4_us=t_rtn, seedlm_us=t_sd,
                             int4_speedup=t_bf / t_rtn, seedlm_speedup=t_bf / t_sd,
                             int4_cfg=c_rtn, seedlm_cfg=c_sd,
                             bf16_GBs=R * N * 2 / t_bf / 1e3, seedlm_GBs=R * N * 0.5 / t_sd / 1e3))
        del Wb, wr, words
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    print("saved", a.out, "| GPU:", torch.cuda.get_device_name(dev))


if __name__ == "__main__":
    main()
