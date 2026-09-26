import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from equations import flops, bytes_moved

RESULTS_DIR = Path(__file__).parent / "results"


def report_collinearity(F, Bm):
    """FLOPs(S,B) and Bytes(S,B) are both ~ S^2*B, so on this grid they are
    nearly perfectly correlated. Plain lstsq can then push one coefficient
    negative (e.g. a negative 's per byte') even though the overall fit
    looks fine -- report the correlation so it ends up in the writeup.
    """
    corr = np.corrcoef(F, Bm)[0, 1]
    print(f"corr(FLOPs, Bytes) on training grid = {corr:.10f}  "
          f"(near 1.0 => multicollinear; that's why we use nnls, not lstsq)")


def fit_latency(df_train):
    F = flops(df_train.S.values, df_train.B.values)
    Bm = bytes_moved(df_train.S.values, df_train.B.values)
    report_collinearity(F, Bm)
    X = np.column_stack([np.ones_like(F), F, Bm])
    y = df_train.latency_s.values
    # theta0/1/2 are physically a launch overhead and inverse throughputs --
    # all >= 0. nnls enforces that instead of letting a collinear lstsq fit
    # push one of them negative.
    theta, _residual = nnls(X, y)
    return theta  # theta0, theta1, theta2


def fit_energy(df_train, latency_theta):
    F = flops(df_train.S.values, df_train.B.values)
    Bm = bytes_moved(df_train.S.values, df_train.B.values)
    theta0, theta1, theta2 = latency_theta
    latency_pred = theta0 + theta1 * F + theta2 * Bm
    X = np.column_stack([latency_pred, F, Bm])
    y = df_train.energy_j.values
    theta, _residual = nnls(X, y)
    p_idle, theta3, theta4 = theta
    return p_idle, theta3, theta4


def relative_error(y_true, y_pred):
    return np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), 1e-12)


def main():
    df = pd.read_csv(RESULTS_DIR / "measurements.csv")
    df = df[df.oom == 0].dropna(subset=["latency_s", "memory_bytes"])

    df_train = df[df.is_validation == 0]
    df_val = df[df.is_validation == 1]

    latency_theta = fit_latency(df_train)
    print(f"latency theta (theta0, theta1, theta2) = {latency_theta.tolist()}")

    F_val = flops(df_val.S.values, df_val.B.values)
    Bm_val = bytes_moved(df_val.S.values, df_val.B.values)
    lat_pred_val = latency_theta[0] + latency_theta[1] * F_val + latency_theta[2] * Bm_val
    lat_err = relative_error(df_val.latency_s.values, lat_pred_val)
    print(f"latency: median relative error on held-out (S,B) = {np.median(lat_err):.3f}")

    theta_out = {
        "latency_theta0": float(latency_theta[0]),
        "latency_theta1": float(latency_theta[1]),
        "latency_theta2": float(latency_theta[2]),
    }

    if df_train.energy_j.notna().any():
        df_train_e = df_train.dropna(subset=["energy_j"])
        p_idle, theta3, theta4 = fit_energy(df_train_e, latency_theta)
        print(f"energy theta (p_idle, theta3, theta4) = {[p_idle, theta3, theta4]}")

        df_val_e = df_val.dropna(subset=["energy_j"])
        if len(df_val_e):
            F_val_e = flops(df_val_e.S.values, df_val_e.B.values)
            Bm_val_e = bytes_moved(df_val_e.S.values, df_val_e.B.values)
            lat_pred_val_e = (latency_theta[0] + latency_theta[1] * F_val_e
                               + latency_theta[2] * Bm_val_e)
            e_pred_val = p_idle * lat_pred_val_e + theta3 * F_val_e + theta4 * Bm_val_e
            e_err = relative_error(df_val_e.energy_j.values, e_pred_val)
            print(f"energy: median relative error on held-out (S,B) = {np.median(e_err):.3f}")

        theta_out.update({
            "p_idle": float(p_idle),
            "energy_theta3": float(theta3),
            "energy_theta4": float(theta4),
        })
    else:
        print("no energy measurements found (pynvml unavailable?) -- skipped energy fit")

    out_path = RESULTS_DIR / "theta.json"
    with open(out_path, "w") as f:
        json.dump(theta_out, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
