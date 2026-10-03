import copy

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import awq

assert torch.cuda.is_available(), "GPU only (no CPU fallback)"
DEV = "cuda:0"


def tiny_model():
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256)
    return Qwen3ForCausalLM(cfg).to(DEV).eval()


def run(bits, dtype):
    model = tiny_model().to(dtype)
    ids = torch.randint(0, 512, (2, 64), device=DEV)
    ref = model(ids).logits.float()
    awq.calib_ids = lambda tok, n, seqlen, source: torch.randint(0, 512, (n, seqlen), device=DEV)
    awq.awq_quantize(model, None, bits, 0, n_samples=16, seqlen=64, source="x", max_tokens=1024, device=DEV)
    return ref, model(ids).logits.float()


def test_function_preserved():
    ref, out = run(16, torch.float32)
    rel = ((ref - out).norm() / ref.norm()).item()
    print(f"  16-bit, fp32: relative logit change {rel:.2e}")
    assert rel < 2e-3, rel


def test_4bit_runs():
    ref, out = run(4, torch.float32)
    rel = ((ref - out).norm() / ref.norm()).item()
    print(f"  4-bit, fp32: relative logit change {rel:.3f}")
    assert torch.isfinite(out).all() and rel < 1.0, rel


if __name__ == "__main__":
    test_function_preserved()
    test_4bit_runs()
    print("ok")
