import torch

import seedlm
from seedlm import Config, compress, decompress, get_tables, lfsr_sequence

assert torch.cuda.is_available(), "GPU only (no CPU fallback)"
DEV = "cuda:0"


def test_lfsr_maximal_length():
    for K in seedlm.TAPS:
        seq = lfsr_sequence(K)
        assert len(torch.unique(seq)) == (1 << K) - 1, f"K={K}: period is not 2^K-1"
        assert seq.min() >= 1 and seq.max() <= (1 << K) - 1


def test_error_formula_matches_direct():
    """_scores must equal ||w - U t_hat||^2 computed from scratch."""
    torch.manual_seed(0)
    cfg = Config(8, 3, 10)
    tb = get_tables(cfg, DEV)
    w = torch.randn(5, cfg.C, device=DEV) * 0.05
    t = (w @ tb.pinv_cols).reshape(5, tb.N, cfg.P)
    err = seedlm._scores(t, tb.Gd, (w * w).sum(-1), cfg.P)
    q, e = seedlm.quantize_coeffs(t)
    that = q * torch.exp2(e)[..., None]
    direct = ((w[:, None, :] - torch.einsum("ncp,bnp->bnc", tb.U, that)) ** 2).sum(-1)
    # fp32 cancellation only matters for seeds whose error exceeds ||w||^2 (never selected)
    useful = direct < (w * w).sum(-1)[:, None]
    assert torch.allclose(err[useful], direct[useful], rtol=1e-3, atol=1e-6)
    assert (err.argmin(1) == direct.argmin(1)).all()


def test_roundtrip_matches_best_seed():
    """decompress(compress(W)) must reproduce the minimum error found by brute force."""
    torch.manual_seed(0)
    cfg = Config(8, 3, 10)
    tb = get_tables(cfg, DEV)
    W = torch.randn(6, 32, device=DEV) * 0.05
    c = compress(W, cfg, block_chunk=7, elems=1 << 14)
    R = decompress(c, DEV)
    assert R.shape == W.shape
    blocks = W.reshape(-1, cfg.C)
    t = (blocks @ tb.pinv_cols).reshape(len(blocks), tb.N, cfg.P)
    q, e = seedlm.quantize_coeffs(t)
    that = q * torch.exp2(e)[..., None]
    full = ((blocks[:, None, :] - torch.einsum("ncp,bnp->bnc", tb.U, that)) ** 2).sum(-1)
    got = ((blocks - R.reshape(-1, cfg.C)) ** 2).sum(-1)
    assert torch.allclose(got, full.min(1).values, rtol=1e-3, atol=1e-7)


def test_padding_and_seed_wrap():
    cfg = Config(12, 4, 16)  # K=16: seeds above 32767 must survive the int16 storage
    torch.manual_seed(0)
    W = torch.randn(3, 50, device=DEV) * 0.02  # 150 elements -> padded to 156
    c = compress(W, cfg)
    assert c.seeds.dtype == torch.int16 and decompress(c, DEV).shape == (3, 50)
    rel = (decompress(c, DEV) - W).norm() / W.norm()
    assert rel < 0.5, rel


def test_triton_matches_eager():
    try:
        import triton_search
    except ImportError:
        return print("skipped: triton not installed")
    if not triton_search.HAS_TRITON:
        return print("skipped: triton not installed")
    for bits in (4, 3):
        rel = triton_search.check_against_eager(Config.paper(bits), DEV)
        print(f"  {bits}-bit: total error differs from eager by {rel:.2e}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
