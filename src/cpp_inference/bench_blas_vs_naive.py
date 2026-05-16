#!/usr/bin/env python3
"""
Standalone correctness + latency check for fp32_conv2d_blas vs fp32_conv2d.

Compares the new BLAS-backed conv against the naive C nested-loop conv and
PyTorch's F.conv2d reference, across the layer shapes that ResNet18 uses on
CIFAR-32 input.

Methodology:
- Pin BLAS / OpenMP / MKL to 1 thread so the comparison is single-thread
  on both sides (the OpenMP loops inside xnor_kernel.so already default to
  1 thread via constructor). Without this, libopenblas would spread sgemm
  across cores while our naive loops stay single-thread, inflating the
  BLAS speedup unfairly.
- Warmup 3 iters before timing.
- Take the min of 30 timed iters (most stable measure; mean is biased high
  by occasional scheduling jitter).

Usage:  python3 bench_blas_vs_naive.py
"""
# Must set BEFORE ctypes.CDLL — BLAS reads these on dlopen.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import ctypes
import time

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)

DIR = os.path.dirname(os.path.abspath(__file__))
xnor_lib = ctypes.CDLL(os.path.join(DIR, "xnor_kernel.so"))
blas_lib = ctypes.CDLL(os.path.join(DIR, "fp32_blas_kernel.so"))

# Force xnor_kernel.so OpenMP pool to 1 thread (its constructor already sets 1,
# but be explicit so any env override doesn't accidentally enable threading).
xnor_lib.cpu_set_num_threads.argtypes = [ctypes.c_int]
xnor_lib.cpu_set_num_threads.restype  = None
xnor_lib.cpu_set_num_threads(1)

# Signature for both implementations is identical.
for fn in (xnor_lib.fp32_conv2d, blas_lib.fp32_conv2d_blas):
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),   # input
        ctypes.POINTER(ctypes.c_float),   # weight
        ctypes.POINTER(ctypes.c_float),   # bias (or NULL)
        ctypes.c_int, ctypes.c_int, ctypes.c_int,                 # C_in, H, W
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,   # C_out, K, stride, padding
        ctypes.POINTER(ctypes.c_float),   # output
    ]
    fn.restype = None


def fptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def call_conv(fn, x, w, b, C_in, H, W, C_out, K, stride, padding):
    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1
    out = np.zeros((C_out, H_out, W_out), dtype=np.float32)
    fn(fptr(x), fptr(w), fptr(b) if b is not None else ctypes.cast(0, ctypes.POINTER(ctypes.c_float)),
       C_in, H, W, C_out, K, stride, padding, fptr(out))
    return out


def torch_conv(x, w, b, stride, padding):
    xt = torch.from_numpy(x).unsqueeze(0)
    wt = torch.from_numpy(w)
    bt = torch.from_numpy(b) if b is not None else None
    return F.conv2d(xt, wt, bt, stride=stride, padding=padding).squeeze(0).numpy()


def _time_min_ms(fn, warmup=3, repeats=30):
    """Return min latency in ms (most stable; mean is biased high by jitter)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times) * 1000.0


def bench_one(name, C_in, H, W, C_out, K, stride, padding, repeats=30):
    np.random.seed(42)
    x = np.random.randn(C_in, H, W).astype(np.float32)
    w = np.random.randn(C_out, C_in, K, K).astype(np.float32) * 0.1
    b = np.random.randn(C_out).astype(np.float32) * 0.01

    out_naive = call_conv(xnor_lib.fp32_conv2d,        x, w, b, C_in, H, W, C_out, K, stride, padding)
    out_blas  = call_conv(blas_lib.fp32_conv2d_blas,   x, w, b, C_in, H, W, C_out, K, stride, padding)
    out_torch = torch_conv(x, w, b, stride, padding)

    diff_naive_torch = np.max(np.abs(out_naive - out_torch))
    diff_blas_torch  = np.max(np.abs(out_blas  - out_torch))

    t_naive = _time_min_ms(
        lambda: call_conv(xnor_lib.fp32_conv2d, x, w, b, C_in, H, W, C_out, K, stride, padding),
        warmup=3, repeats=repeats,
    )
    t_blas = _time_min_ms(
        lambda: call_conv(blas_lib.fp32_conv2d_blas, x, w, b, C_in, H, W, C_out, K, stride, padding),
        warmup=3, repeats=repeats,
    )

    speedup = t_naive / max(t_blas, 1e-6)
    print(f"  {name:<22} naive {t_naive:8.3f} ms   BLAS {t_blas:7.3f} ms   "
          f"speedup {speedup:5.2f}× | "
          f"diff vs torch: naive {diff_naive_torch:.2e}, BLAS {diff_blas_torch:.2e}")
    return t_naive, t_blas, diff_blas_torch


# ResNet18 CIFAR-32 layer shapes (after stem-adapt, no maxpool):
layers = [
    ("stem 3->64",        3,   32, 32,  64, 3, 1, 1),
    ("layer1 64->64",     64,  32, 32,  64, 3, 1, 1),
    ("layer2 64->128/s2", 64,  32, 32, 128, 3, 2, 1),
    ("layer2 128->128",   128, 16, 16, 128, 3, 1, 1),
    ("layer3 128->256/s2",128, 16, 16, 256, 3, 2, 1),
    ("layer3 256->256",   256,  8,  8, 256, 3, 1, 1),
    ("layer4 256->512/s2",256,  8,  8, 512, 3, 2, 1),
    ("layer4 512->512",   512,  4,  4, 512, 3, 1, 1),
    ("downsample 1x1/s2", 64,  32, 32, 128, 1, 2, 0),
]


print(f"{'Layer':<24} {'naive (ms)':>12} {'BLAS (ms)':>11} {'speedup':>9}   accuracy gap")
print("-" * 95)
total_naive = total_blas = 0
max_blas_diff = 0.0
for spec in layers:
    n, b, d = bench_one(*spec)
    total_naive += n
    total_blas  += b
    max_blas_diff = max(max_blas_diff, d)

print("-" * 95)
print(f"  {'TOTAL':<22} naive {total_naive:8.3f} ms   BLAS {total_blas:7.3f} ms   "
      f"speedup {total_naive/total_blas:5.2f}× | max BLAS-vs-torch diff: {max_blas_diff:.2e}")
