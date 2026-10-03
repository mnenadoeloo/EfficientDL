"""SeedLM (Shafipour et al., arXiv:2410.10714): weights -> (LFSR seed, 4-bit coefficients)."""
from dataclasses import dataclass
from functools import lru_cache

import torch

TAPS = {3: (3, 2), 4: (4, 3), 5: (5, 3), 6: (6, 5), 7: (7, 6), 8: (8, 6, 5, 4), 9: (9, 5),
        10: (10, 7), 11: (11, 9), 12: (12, 11, 10, 4), 13: (13, 12, 11, 8),
        14: (14, 13, 12, 2), 15: (15, 14), 16: (16, 15, 13, 4)}

PAPER_CONFIGS = {4: (8, 3, 16), 3: (12, 4, 16)}

COND_MAX = 100
EXP_MIN, EXP_MAX = -8, 7 
Q_MIN, Q_MAX = -8, 7 


def lfsr_sequence(K: int) -> torch.Tensor:
    """All 2^K - 1 consecutive states of the K-bit LFSR, starting from state 1."""
    taps, mask = TAPS[K], (1 << K) - 1
    state, out = 1, []
    for _ in range(mask):
        out.append(state)
        fb = 0
        for t in taps:
            fb ^= (state >> (t - 1)) & 1
        state = ((state << 1) | fb) & mask
    return torch.tensor(out, dtype=torch.int64)


@dataclass(frozen=True)
class Config:
    C: int
    P: int
    K: int

    @property
    def bits_per_weight(self) -> float:
        return (self.K + 4 + 4 * self.P) / self.C

    @staticmethod
    def paper(bits: int, K: int = None) -> "Config":
        C, P, K0 = PAPER_CONFIGS[bits]
        return Config(C, P, K0 if K is None else K)


class Tables:
    """LFSR state cycle and, per candidate seed, U, pinv(U) and the Gram matrix U^T U."""

    def __init__(self, cfg: Config, device="cpu"):
        self.cfg, self.device = cfg, torch.device(device)
        K, C, P = cfg.K, cfg.C, cfg.P
        self.N = (1 << K) - 1
        seq = lfsr_sequence(K)
        self.seq = seq.to(self.device)
        pos = torch.zeros(1 << K, dtype=torch.int64)
        pos[seq] = torch.arange(self.N)
        self.pos = pos.to(self.device)
        U = self.U_from_index(torch.arange(self.N, device=self.device)) # (N, C, P)
        U64 = U.double()
        G = U64.transpose(1, 2) @ U64 # (N, P, P)
        pinv = torch.linalg.pinv(U64) # (N, P, C)
        bad = torch.linalg.cond(U64) > COND_MAX
        self.n_disabled = int(bad.sum())
        pinv[bad] = 0
        self.pinv_cols = pinv.permute(2, 0, 1).reshape(C, self.N * P).float().contiguous()
        self.pinv = pinv.float()
        self.U = U
        self.Gd = G.float().permute(1, 2, 0).contiguous() # (P, P, N)

    def U_from_index(self, j: torch.Tensor) -> torch.Tensor:
        """U for the seed that is the j-th state of the cycle. V is filled starting with the
        value generated after the seed state (not the seed itself), row-major into C x P."""
        C, P, K = self.cfg.C, self.cfg.P, self.cfg.K
        idx = (j[:, None] + 1 + torch.arange(C * P, device=j.device)[None]) % self.N
        V = self.seq[idx].reshape(-1, C, P).float()
        return (V - (1 << (K - 1))) / ((1 << (K - 1)) - 1)  # Eq. (1)

    def U_from_seed(self, s: torch.Tensor) -> torch.Tensor:
        return self.U_from_index(self.pos[s.long()])


@lru_cache(maxsize=8)
def get_tables(cfg: Config, device: str) -> Tables:
    return Tables(cfg, device)


def quantize_coeffs(t: torch.Tensor):
    """t: (..., P) -> (q int in [-8,7], e exponent in [-8,7]) with t ~= q * 2^e.

    The shared exponent is taken from the largest |t_i| (the paper's e = max floor(log2|t_i|)),
    shifted by 2 so that the largest coefficient lands in [4, 8) and uses the 4-bit integer range.
    """
    amax = t.abs().amax(-1, keepdim=True)
    e = (torch.floor(torch.log2(amax)) - 2).clamp(EXP_MIN, EXP_MAX)
    q = torch.round(t / torch.exp2(e)).clamp(Q_MIN, Q_MAX)
    return q, e.squeeze(-1)


def _quad(x: torch.Tensor, Gd: torch.Tensor, P: int) -> torch.Tensor:
    """x^T G x for every (block, seed). x: (nb, Ns, P), Gd: (P, P, Ns)."""
    out = 0
    for p in range(P):
        out = out + Gd[p, p] * x[..., p] ** 2
        for q in range(p + 1, P):
            out = out + 2 * Gd[p, q] * x[..., p] * x[..., q]
    return out


def _scores(t: torch.Tensor, Gd: torch.Tensor, wn2: torch.Tensor, P: int) -> torch.Tensor:
    """Reconstruction error ||w - U t_hat||^2 for the quantized least-squares coefficients.

    t is the least-squares solution, so w - U t is orthogonal to range(U) and
    ||w - U t_hat||^2 = ||w||^2 - t^T G t + (t_hat - t)^T G (t_hat - t)  with G = U^T U.
    This is exactly the paper's eps_j but only needs P-dimensional tensors.
    """
    q, e = quantize_coeffs(t)
    d = q * torch.exp2(e)[..., None] - t
    return wn2[:, None] - _quad(t, Gd, P) + _quad(d, Gd, P)


_scores_compiled = None


def _get_scores(use_compile: bool):
    global _scores_compiled
    if not use_compile:
        return _scores
    if _scores_compiled is None:
        _scores_compiled = torch.compile(_scores, dynamic=True)
    return _scores_compiled


@dataclass
class Compressed:
    shape: tuple
    cfg: Config
    seeds: torch.Tensor # (nb,) int16 holding the K-bit seed (wrapped, mask with 0xFFFF)
    exps: torch.Tensor # (nb,) int8
    q: torch.Tensor # (nb, P) int8


@torch.no_grad()
def compress(W: torch.Tensor, cfg: Config, tables: Tables = None, block_chunk: int = 4096,
             elems: int = 1 << 25, use_compile: bool = False, kernel: str = "eager") -> Compressed:
    """Algorithm 1 for every block of W (flattened row-major, zero padded to a multiple of C).

    kernel="eager" is the plain PyTorch search; kernel="triton" is the fused kernel of
    triton_search.py (same result, much faster, CUDA only)."""
    tb = tables or get_tables(cfg, str(W.device))
    C, P, N = cfg.C, cfg.P, tb.N
    flat = W.detach().float().reshape(-1)
    nb = -(-flat.numel() // C)
    blocks = torch.nn.functional.pad(flat, (0, nb * C - flat.numel())).reshape(nb, C)
    if kernel == "triton":
        import triton_search
        best_j = triton_search.search(blocks, tb)
    else:
        ns_chunk = max(256, min(N, elems // (block_chunk * P)))
        score_fn = _get_scores(use_compile)
        best_j = torch.empty(nb, dtype=torch.int64, device=W.device)
        for b0 in range(0, nb, block_chunk):
            w = blocks[b0:b0 + block_chunk]
            wn2 = (w * w).sum(-1)
            best_e = torch.full((w.shape[0],), float("inf"), device=W.device)
            bj = torch.zeros(w.shape[0], dtype=torch.int64, device=W.device)
            for s0 in range(0, N, ns_chunk):
                s1 = min(N, s0 + ns_chunk)
                t = (w @ tb.pinv_cols[:, s0 * P:s1 * P]).reshape(w.shape[0], s1 - s0, P)
                err = score_fn(t, tb.Gd[:, :, s0:s1], wn2, P)
                m, a = err.min(1)
                upd = m < best_e
                best_e = torch.where(upd, m, best_e)
                bj = torch.where(upd, a + s0, bj)
            best_j[b0:b0 + w.shape[0]] = bj
    t = torch.einsum("bpc,bc->bp", tb.pinv[best_j], blocks)
    q, e = quantize_coeffs(t)
    seeds = tb.seq[best_j].to(torch.int32).to(torch.int16)
    return Compressed(tuple(W.shape), cfg, seeds, e.to(torch.int8), q.to(torch.int8))


@torch.no_grad()
def decompress(c: Compressed, device="cpu", chunk: int = 1 << 20) -> torch.Tensor:
    """Rebuild the weights: regenerate U(s) from the seed with the LFSR and multiply by t."""
    cfg = c.cfg
    tb = get_tables(cfg, str(torch.device(device)))
    out = []
    for b0 in range(0, c.seeds.shape[0], chunk):
        s = c.seeds[b0:b0 + chunk].to(device).to(torch.int32) & 0xFFFF
        t = c.q[b0:b0 + chunk].to(device).float() * torch.exp2(c.exps[b0:b0 + chunk].to(device).float())[:, None]
        out.append(torch.einsum("bcp,bp->bc", tb.U_from_seed(s), t))
    numel = 1
    for d in c.shape:
        numel *= d
    return torch.cat(out).reshape(-1)[:numel].reshape(c.shape)
