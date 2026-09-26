import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from equations import flops, bytes_moved, memory as memory_eq

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GRID_KW = dict(color="#c9c9c9", linewidth=0.6, alpha=0.6)


def load():
    df = pd.read_csv(RESULTS_DIR / "measurements.csv")
    theta = json.loads((RESULTS_DIR / "theta.json").read_text())
    return df, theta


def latency_pred(S, B, theta):
    return (theta["latency_theta0"]
            + theta["latency_theta1"] * flops(S, B)
            + theta["latency_theta2"] * bytes_moved(S, B))


def energy_pred(S, B, theta):
    if "p_idle" not in theta:
        return None
    t = latency_pred(S, B, theta)
    return (theta["p_idle"] * t
            + theta["energy_theta3"] * flops(S, B)
            + theta["energy_theta4"] * bytes_moved(S, B))


def savefig(fig, name):
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def plot_vs_S(df, theta, value_col, pred_fn, fixed_Bs, ylabel, title, fname, log_y=True):
    ok = df[df.oom == 0].dropna(subset=[value_col])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    S_line = np.linspace(32, 512, 200)

    for i, B in enumerate(fixed_Bs):
        color = COLORS[i % len(COLORS)]
        sub = ok[ok.B == B].sort_values("S")
        if len(sub):
            ax.scatter(sub.S, sub[value_col], color=color, s=28, zorder=3,
                       label=f"B={B}")
        y_line = pred_fn(S_line, B, theta)
        ax.plot(S_line, y_line, color=color, linewidth=1.8, alpha=0.9)

    ax.set_xlabel("S (image side, px)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(True, which="both", **GRID_KW)
    ax.legend(fontsize=8, title="dots = measured, line = model")
    savefig(fig, fname)


def plot_parity(df, theta, value_col, pred_fn, ylabel, title, fname):
    ok = df[df.oom == 0].dropna(subset=[value_col])
    S = ok.S.values
    B = ok.B.values
    y_meas = ok[value_col].values
    y_pred = pred_fn(S, B, theta)

    is_val = ok.is_validation.values.astype(bool)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_meas[~is_val], y_pred[~is_val], color=COLORS[0], s=24,
               label="base grid", alpha=0.85)
    ax.scatter(y_meas[is_val], y_pred[is_val], color=COLORS[1], s=24,
               marker="^", label="validation", alpha=0.85)

    lo = min(y_meas.min(), y_pred.min())
    hi = max(y_meas.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="#8a8a8a", linewidth=1.2, linestyle="--",
             label="y = x")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"measured, {ylabel}")
    ax.set_ylabel(f"predicted, {ylabel}")
    ax.set_title(title)
    ax.grid(True, which="both", **GRID_KW)
    ax.legend(fontsize=8)
    savefig(fig, fname)


def plot_memory_with_oom(df, fixed_Bs):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    S_line = np.linspace(32, 512, 200)

    oom_by_B = {}
    y_max_data = 0.0
    for i, B in enumerate(fixed_Bs):
        color = COLORS[i % len(COLORS)]
        sub_ok = df[(df.B == B) & (df.oom == 0)].dropna(subset=["memory_bytes"]).sort_values("S")
        oom_by_B[B] = df[(df.B == B) & (df.oom == 1)]

        if len(sub_ok):
            ax.scatter(sub_ok.S, sub_ok.memory_bytes / 1e9, color=color, s=28,
                       zorder=3, label=f"B={B}")
            y_max_data = max(y_max_data, (sub_ok.memory_bytes / 1e9).max())

        y_line = memory_eq(S_line, B) / 1e9
        ax.plot(S_line, y_line, color=color, linewidth=1.8, alpha=0.9)
        y_max_data = max(y_max_data, y_line.max())

    oom_y = y_max_data * 1.05 if y_max_data > 0 else 1.0
    oom_seen = False
    for i, B in enumerate(fixed_Bs):
        color = COLORS[i % len(COLORS)]
        sub_oom = oom_by_B[B]
        if len(sub_oom):
            ax.scatter(sub_oom.S, [oom_y] * len(sub_oom), color=color,
                       s=60, marker="x", zorder=4,
                       label=None if oom_seen else "OOM")
            oom_seen = True

    ax.set_xlabel("S (image side, px)")
    ax.set_ylabel("Memory, GB")
    ax.set_title("Memory(S,B)")
    ax.set_xscale("log")
    ax.grid(True, which="both", **GRID_KW)
    ax.legend(fontsize=8, title="dots = measured, line = model")
    savefig(fig, "memory_vs_S.png")


def plot_regimes(df, theta):
    ok = df[df.oom == 0].dropna(subset=["latency_s"])
    S = ok.S.values
    B = ok.B.values
    work = flops(S, B)
    lat = ok.latency_s.values

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(work, lat, color=COLORS[0], s=18, alpha=0.7, label="measured")

    order = np.argsort(work)
    work_s, lat_s = work[order], lat[order]
    ax.plot(work_s, lat_s, color=COLORS[0], linewidth=0.8, alpha=0.3)

    launch = np.full_like(work_s, theta["latency_theta0"], dtype=float)
    ax.axhline(theta["latency_theta0"], color="#8a8a8a", linewidth=1.2, linestyle="--",
               label=f"launch overhead θ0={theta['latency_theta0']*1e3:.3f} ms")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FLOPs(S,B)")
    ax.set_ylabel("Latency, s")
    ax.set_title("Regimes: launch-bound -> memory/compute-bound")
    ax.grid(True, which="both", **GRID_KW)
    ax.legend(fontsize=8)
    savefig(fig, "regimes.png")


def main():
    df, theta = load()
    fixed_Bs = [b for b in [1, 16, 256] if b in df.B.values]
    if not fixed_Bs:
        fixed_Bs = sorted(df.B.unique())[:3]

    plot_vs_S(df, theta, "latency_s", latency_pred, fixed_Bs,
               ylabel="Latency, s", title="Latency(S,B)",
               fname="latency_vs_S.png")

    plot_parity(df, theta, "latency_s", latency_pred,
                ylabel="s", title="Latency: parity plot",
                fname="latency_parity.png")

    plot_memory_with_oom(df, fixed_Bs)

    plot_parity(df, theta, "memory_bytes",
                lambda S, B, th: memory_eq(S, B),
                ylabel="bytes", title="Memory: parity plot",
                fname="memory_parity.png")

    plot_regimes(df, theta)

    if "p_idle" in theta and df.energy_j.notna().any():
        plot_vs_S(df, theta, "energy_j", energy_pred, fixed_Bs,
                   ylabel="Energy, J", title="Energy(S,B)",
                   fname="energy_vs_S.png")
        plot_parity(df, theta, "energy_j", energy_pred,
                    ylabel="J", title="Energy: parity plot",
                    fname="energy_parity.png")
    else:
        print("energy_j missing from data or theta has no energy fields -- skipping energy plots")


if __name__ == "__main__":
    main()
