# Homework 1 — Analytical performance model of a small CNN

**Issued:** 19.09.2026 · **Due:** 26.09.2026 · **Points:** 5

## Submission link:
https://forms.gle/itnp6pXrZi5TsD5H6

---

## 1. Goal

Take a small convolutional network and describe its cost. You will derive four closed-form functions of the input image size $S$ and the batch size $B$:

| Function | What it predicts | Depends on calibrated parameters? |
|----------|------------------|-----------------------------------|
| `FLOPs(S, B)` | floating-point operations of one forward pass | no |
| `Memory(S, B)` | peak GPU memory of one forward pass | no  |
| `Latency(S, B, θ)` | wall-clock time of one forward pass | yes — $\theta$ is calibrated on your GPU |
| `Energy(S, B, θ)` | energy of one forward pass | yes — $\theta$ is calibrated on your GPU |

Measure the real network on a real GPU, calibrate $\theta$, check how well the equations predict *unseen* $(S, B)$ points, and explain where and why they break. 
The interesting part is the breaking: **launch-bound → memory-bound → compute-bound** regimes.

## 2. Hardware

Everything should run on **one low-grade GPU**, such as the free tier of **Google Colab** (typically a Tesla T4) or **Kaggle** (P100 or T4). 
A local NVIDIA GPU with 16 GB or less is fine too.

## 3. Models

Choose the  model below. All are sequential — no residual connections — and are built only from convolution, BatchNorm, ReLU, pooling and linear layers. All run in `eval()` mode, FP32. Every convolution uses `padding = k // 2` and `bias=False`; every ReLU is `inplace=True`; input is `3 × S × S`; 100 output classes. After each convolution layer we use ReLU.


**Model** 
| Layers | Output resolution |
|--------|-------------------|
| Conv7×7 s2 3→32, MaxPool 3×3 s2 p1 | S/4 |
| Conv5×5 32→64 | S/4 |
| Conv3×3 s2 64→128 | S/8 |
| Conv1×1 128→256 | S/8 |
| Conv3×3 s2 256→256 | S/16 |
| Conv1×1 256→512 | S/16 |
| head: GlobalAvgPool, Linear 512→256, ReLU, Linear 256→100 | — |


## 4. Measurement grid

Image sizes are square, $H = W = S$, always a multiple of 16 (so every stride-2 layer divides evenly).

| | Base grid | Randomly sampled (in addition) |
|---|-----------|--------------------------------|
| Image size $S$ | 32, 64, 128, 224, 256, 384, 512 | 4 multiples of 16 in [32, 512], not in the base grid |
| Batch size $B$ | 1, 2, 4, 8, 16, 32, 64, 128, 256 | 3 integers in [1, 256], not powers of two |


Do not use real images, use random tensors as inputs.

That gives 11 image sizes × 12 batch sizes = 132 configurations. 

Some of them (large $S$ with large $B$) will not fit into GPU memory — that is **expected**. 

Catch the `torch.cuda.OutOfMemoryError`, record the configuration as `OOM`, and compare it with what your `Memory` equation predicts.


## 5. Derivations on paper (handwritten)

Derive, for **your** network, the four functions as explicit formulas in $S$ and $B$. For each function:

State your assumptions for latency and energy explicitly (see the conventions below).

**Conventions to state and follow**

- **FLOPs.** 1 multiply-accumulate = 2 FLOPs. 
- **Memory.** The peak of `torch.cuda.max_memory_allocated()` during one forward pass under `torch.inference_mode()`. 
- **Bytes moved** (needed for latency and energy).
- **Latency.** Median time of one forward pass in `eval()` + `inference_mode()`, FP32.
- **Energy.** Energy of one forward pass, in joules, measured on the whole GPU.


## 6. Code

Implement your equations as plain Python functions that take numbers and return numbers:

```python
def flops(image_size, batch): ...                  # -> float (FLOPs)
def memory(image_size, batch): ...                 # -> float (bytes)
def latency(image_size, batch, theta): ...         # -> float (seconds)
def energy(image_size, batch, theta_energy): ...   # -> float (joules)
```

They must accept NumPy arrays for `image_size` and `batch` (broadcasting), so that plotting a surface over the $(S, B)$ plane is a single call.


## 7. Measurement protocol

Set these flags first and keep them the same in all runs:

```python
torch.backends.cudnn.benchmark = False        # main results: PyTorch's default heuristics choose the kernel
torch.backends.cudnn.allow_tf32 = False       # so that FP32 means FP32 on Ampere and newer GPUs
torch.backends.cuda.matmul.allow_tf32 = False
model = model.cuda().eval()
```


## 8. Plots and analysis

Validate the equations against the real network - plot predicted versus real results as the grid above.
Every plot shows **measured points and the predicted curve/surface together**, with labelled axes and units.

## 10. Deliverables

Submit a **link to your GitHub repository** containing everything below. 

```
hw1/
├── README.md                # GPU / software versions, how to reproduce, results summary, 1-page discussion
├── hw1_handwritten.pdf      # scanned or photographed handwritten derivations (one PDF, legible)
├── models.py                # your network
├── equations.py             # flops(), memory(), latency(), energy()
├── measure.py               # measurements (or a notebook that runs on Colab/Kaggle end to end)
├── calibrate.py             # fitting of theta
└── results/
    ├── measurements.csv     # S, B, latency, memory or OOM, energy, is_validation
    ├── theta.json           # fitted parameters
    └── figures/*.png
```