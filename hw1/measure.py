import csv
import random
import time
from pathlib import Path

import numpy as np
import torch

from models import build_model

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

DEVICE = "cuda"
N_REPEATS = 21
N_WARMUP = 5
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def build_grid():
    base_S = [32, 64, 128, 224, 256, 384, 512]
    base_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    rng = random.Random(0)

    def sample_S(n):
        pool = [s for s in range(32, 513, 16) if s not in base_S]
        return rng.sample(pool, n)

    def sample_B(n):
        pool = [b for b in range(1, 257) if b not in base_B and (b & (b - 1)) != 0]
        return rng.sample(pool, n)

    extra_S = sample_S(4)
    extra_B = sample_B(3)

    configs = []
    for S in base_S + extra_S:
        for B in base_B + extra_B:
            is_validation = (S in extra_S) or (B in extra_B)
            configs.append((S, B, is_validation))
    return configs


def try_nvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml, handle
    except Exception:
        return None, None


def measure_energy_joules(pynvml_mod, handle, fn, n_repeats):
    """Sample instantaneous power (mW) around the timed loop and integrate.
    Coarse but requires no extra dependency beyond pynvml.
    """
    powers = []
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        fn()
        powers.append(pynvml_mod.nvmlDeviceGetPowerUsage(handle) / 1000.0)  # W
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    if not powers:
        return None
    avg_power_w = sum(powers) / len(powers)
    return avg_power_w * elapsed / n_repeats  # J per forward pass


@torch.inference_mode()
def measure_one(model, S, B):
    x = torch.randn(B, 3, S, S, device=DEVICE)

    for _ in range(N_WARMUP):
        model(x)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(N_REPEATS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    latency_s = float(np.median(times))
    memory_bytes = torch.cuda.max_memory_allocated()

    pynvml_mod, handle = try_nvml()
    energy_j = None
    if pynvml_mod is not None:
        energy_j = measure_energy_joules(pynvml_mod, handle, lambda: model(x), n_repeats=11)

    del x
    torch.cuda.empty_cache()
    return latency_s, memory_bytes, energy_j


def main():
    assert torch.cuda.is_available(), "run this on a GPU runtime (Colab/Kaggle)"
    model = build_model().to(DEVICE).eval()

    configs = build_grid()
    out_path = RESULTS_DIR / "measurements.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["S", "B", "latency_s", "memory_bytes", "energy_j",
                          "oom", "is_validation"])

        for i, (S, B, is_val) in enumerate(configs):
            try:
                latency_s, memory_bytes, energy_j = measure_one(model, S, B)
                writer.writerow([S, B, latency_s, memory_bytes, energy_j, 0, int(is_val)])
                print(f"[{i+1}/{len(configs)}] S={S} B={B} "
                      f"latency={latency_s*1e3:.2f}ms mem={memory_bytes/1e6:.1f}MB")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                writer.writerow([S, B, "", "", "", 1, int(is_val)])
                print(f"[{i+1}/{len(configs)}] S={S} B={B} -> OOM")
            f.flush()

    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
