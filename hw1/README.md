# HW1 — Analytical performance model of SmallCNN

## GPU / software

- **GPU:** NVIDIA H100 PCIe, 80 GB (use of H100 instead of a low-grade GPU cleared with the instructor)
- **Driver:** 580.119.02, **CUDA:** 13.0 (from `nvidia-smi`)
- **Python:** 3.12.3, **PyTorch:** 2.14.0+cu130, **CUDA toolkit (PyTorch build):** 13.0

## How to reproduce

```bash
cd hw1
pip install torch numpy pandas scipy matplotlib pynvml  # pynvml optional, needed only for energy
python measure.py      
python calibrate.py  
python plot.py       
```

`equations.py` exposes the four closed-form functions (`flops`, `memory`,
`latency`, `energy`) used both by `calibrate.py` and `plot.py`.

## Results summary

Grid: 7 base + 4 random $S$ values $\times$ 9 base + 3 random $B$ values = 132
configurations, random-tensor inputs, `eval()` + `inference_mode()`, FP32,
`cudnn.benchmark=False`, TF32 disabled (section 7 flags). 0/132 OOM (expected
on an 80 GB GPU - the Memory formula predicts ~3 GB even at the largest
config).

| Function | Calibrated? | Median relative error (held-out validation points) |
|---|---|---|
| FLOPs($S,B$) | no (pure count) | n/a - not directly measured |
| Memory($S,B$) | no (pure count) | systematic **under**-estimate, ~30–40% at large $S,B$ |
| Latency($S,B,\theta$) | yes (nnls) | **8%** |
| Energy($S,B,\theta$) | yes (nnls) | **16%** |

Fitted $\theta$ (`results/theta.json`):

```json
{
  "latency_theta0": 3.23e-4, // s, launch/dispatch overhead
  "latency_theta1": 4.35e-14, // s/FLOP
  "latency_theta2": 0.0, // s/byte
  "p_idle": 0.0, // W
  "energy_theta3": 1.50e-11, // J/FLOP
  "energy_theta4": 0.0 // J/byte
}
```

## Discussion

**FLOPs and Memory** are exact closed-form counts from the architecture: $\text{FLOPs}(S,B) = B(17712\,S^2 + 313344)$,
$\text{Memory}(S,B) \approx 4{,}161{,}296 + 44\,S^2 B$ bytes. No GPU is needed
to derive them, and FLOPs needs no validation against a "ground truth" (there
is nothing to measure it against directly - it is a definition). Memory *is*
measured against `torch.cuda.max_memory_allocated()`, and the formula is a
systematic **lower bound**: it only counts model weights plus the
input+output of the single heaviest layer (conv1), assuming inference-mode
frees everything else. Real measurements run 30–40% higher across the grid -
the gap is cuDNN's convolution-algorithm workspace and PyTorch's caching
allocator (which rounds allocations to block sizes and doesn't return memory
immediately), neither of which the formula accounts for by design.

**Latency and Energy are calibrated, and both expose the same problem.**
$\text{FLOPs}(S,B)$ and $\text{Bytes}(S,B)$ are both $\propto S^2 B$ for this
architecture, so on the measurement grid they are almost perfectly collinear
(Pearson correlation $0.9999999942$). An ordinary least-squares fit exploits
this and pushes one coefficient negative - a negative "seconds per byte" is
not physically meaningful, so calibration uses non-negative least squares
(`scipy.optimize.nnls`) instead. Under near-perfect collinearity, `nnls`
doesn't split weight between the two correlated features; it zeroes one out
entirely. That is exactly what happened: $\theta_2$ (the bytes-moved term)
went to 0 for Latency, and then - because $\text{Latency}_{pred}$ is itself
now an affine function of FLOPs ($\theta_2=0$), the *same* collinearity
recurs one level up in the Energy fit, zeroing $P_{idle}$ too.

$P_{idle}=0$ is a calibration artifact, not a physical claim: raw
`energy_j / latency_s` from the measurements climbs from $\approx$97 W at the
smallest configuration to $\approx$348 W at $S{=}512,B{=}256$, matching the
H100's 350 W TDP under load almost exactly - the underlying measurements are
sound, the model just can't isolate the constant idle-power term given how
this network's FLOPs and Bytes covary.

**The consequence is visible directly in the figures.** `regimes.png` shows a
clean **launch-bound plateau** at $\theta_0 \approx 0.32$ ms for small FLOPs,
then a single power-law growth regime - not the three regimes (launch →
memory → compute) the assignment sets up as the interesting part. Only two
are empirically distinguishable here, because memory-bound and compute-bound
behavior cannot be told apart when Bytes and FLOPs move together. Both
`latency_parity.png` and `energy_parity.png` track the diagonal well at large
$S,B$ (compute-bound regime, where the model is well-identified) and scatter
badly at small $S,B$ (launch-bound regime, where the fitted model
underestimates because it has no genuine constant-overhead term left to
absorb it) - most visibly on `energy_vs_S.png`, where at $S{=}32,B{=}1$ the
model predicts $\approx3\times10^{-4}$ J against a measured $\approx0.02$ J,
roughly 60x off.

**Takeaway:** for this specific sequential architecture, a grid over $(S,B)$
alone cannot separate the launch/idle overhead from the compute-proportional
term, because every configuration moves FLOPs and Bytes in lockstep. Doing so
would require a grid where FLOPs/Bytes ratio (arithmetic intensity) varies
independently of $S^2B$ - not achievable by varying $(S,B)$ on a fixed
sequential conv stack like this one.
