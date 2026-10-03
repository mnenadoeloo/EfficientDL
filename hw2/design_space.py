"""Grid search over (C, P, K) for a fixed bit budget M = (K + 4 + 4P) / C."""
import argparse
import csv
import os

import torch

from seedlm import Config, compress, decompress


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=1024, help="number of Gaussian blocks per config")
    ap.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--kmin", type=int, default=8)
    ap.add_argument("--kmax", type=int, default=16)
    ap.add_argument("--pmax", type=int, default=8)
    ap.add_argument("--out", default="results/design_space.csv")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "GPU only (no CPU fallback)"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows = []
    for M in a.bits:
        for K in range(a.kmin, a.kmax + 1):
            for P in range(1, a.pmax + 1):
                if (K + 4 + 4 * P) % M:
                    continue
                C = (K + 4 + 4 * P) // M
                if C < P + 1:
                    continue
                cfg = Config(C, P, K)
                g = torch.Generator().manual_seed(0)
                W = torch.randn(a.n, C, generator=g).to(a.device)
                err = ((W - decompress(compress(W, cfg), a.device)) ** 2).sum() / (W ** 2).sum()
                rows.append(dict(M=M, C=C, P=P, K=K, rel_err=err.item()))
                print(f"M={M} C={C:2d} P={P} K={K:2d}  rel_err={err.item():.4f}", flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["M", "C", "P", "K", "rel_err"])
        w.writeheader()
        w.writerows(rows)
    for M in a.bits:
        best = min((r for r in rows if r["M"] == M), key=lambda r: r["rel_err"])
        print(f"best for M={M}: C={best['C']} P={best['P']} K={best['K']} rel_err={best['rel_err']:.4f}")


if __name__ == "__main__":
    main()
