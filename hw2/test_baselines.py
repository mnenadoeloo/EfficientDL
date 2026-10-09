"""Smoke tests of omniquant.py, quip_baseline.py and bench_matvec.py on tiny random models (GPU only).

    python test_baselines.py
"""
import os

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import awq

assert torch.cuda.is_available(), "GPU only (no CPU fallback)"
DEV = "cuda:0"
REPO = os.environ.get("QUIP_REPO", "third_party/quip-sharp")


def tiny_model(hidden=64, inter=128):
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=512, hidden_size=hidden, intermediate_size=inter, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=hidden // 4,
                      max_position_embeddings=256)
    return Qwen3ForCausalLM(cfg).to(DEV).eval()


def rel_change(model, quantize):
    ids = torch.randint(0, 512, (2, 64), device=DEV)
    ref = model(ids).logits.float()
    quantize(model)
    out = model(ids).logits.float()
    assert torch.isfinite(out).all()
    return ((out - ref).norm() / ref.norm()).item()


def test_omniquant():
    import omniquant
    omniquant.calib_wikitext = lambda tok, n, seqlen, seed=2: torch.randint(0, 512, (n, seqlen), device=DEV)
    rel = rel_change(tiny_model(), lambda m: omniquant.omniquant(
        m, None, 8, 0, n_samples=4, seqlen=64, epochs=2, device=DEV, log=lambda *a: None))
    print(f"  OmniQuant 8-bit: relative logit change {rel:.3f}")
    assert rel < 0.3, rel


def test_quip():
    if not os.path.isdir(os.path.join(REPO, "lib")):
        return print(f"  skipped: no quip-sharp clone at {REPO}")
    import quip_baseline
    awq.calib_ids = lambda tok, n, seqlen, source: torch.randint(0, 512, (n, seqlen), device=DEV)
    quip_baseline.calib_ids = awq.calib_ids
    for bits in (4, 3):
        rel = rel_change(tiny_model(), lambda m: quip_baseline.quip_quantize(
            m, None, bits, REPO, n_samples=8, seqlen=64, device=DEV, log=lambda *a: None))
        print(f"  QuIP# {bits}-bit: relative logit change {rel:.3f}")
        assert rel < 2.0, rel


def test_had_supported():
    import quip_baseline as q
    assert q.had_supported(4096) and q.had_supported(12288) and q.had_supported(5120)
    assert not q.had_supported(17408) and not q.had_supported(34816)


def test_matvec_kernels():
    import bench_matvec
    from seedlm import Config, compress, decompress, get_tables
    cfg = Config(8, 3, 16)
    tb = get_tables(cfg, DEV)
    W = torch.randn(64, 256, device=DEV) * 0.02
    c = compress(W, cfg, tb)
    words = bench_matvec.pack_seedlm(c, tb)
    x = torch.randn(256, device=DEV).to(torch.bfloat16)
    got = bench_matvec.run_seed(bench_matvec._load_seed_kernel(), words, tb.seq.to(torch.int32), x, (8, 16, 4))
    ref = decompress(c, DEV) @ x.float()
    rel = ((got - ref).norm() / ref.norm()).item()
    print(f"  SeedLM matvec kernel: relative error {rel:.2e}")
    assert rel < 1e-3, rel


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("ok")
