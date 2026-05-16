#!/usr/bin/env python3
"""
3-way latency benchmark — the main presentation chart producer.

Compares for each ResNet18 binary-path layer shape:
  (1) FP32 BLAS sgemm conv         — production FP32 baseline
  (2) Binary conv, scalar popcnt   — what `__builtin_popcountll` alone gives
  (3) Binary conv, AVX-512 VPOPCNTDQ — explicit _mm512_popcnt_epi64 intrinsic

Also runs a GEMM-level micro-bench (xnor_gemm vs xnor_gemm_avx512) so we
can isolate the kernel speedup from the im2col packing + padding correction
overhead that binary_conv2d adds on top.

Methodology (same as bench_blas_vs_naive.py):
- Thread env pinned to 1 for all libraries (set before ctypes.CDLL).
- Warmup 3, time 30, report min ms.
- Bit-exact assertion between scalar and AVX-512 paths for every shape.

Usage:  python3 bench_3way.py
"""
# Pin threads BEFORE ctypes loads BLAS.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import ctypes
import time
from ctypes import c_float, c_int, c_int32, c_uint64, POINTER

import numpy as np


DIR = os.path.dirname(os.path.abspath(__file__))
xnor_lib = ctypes.CDLL(os.path.join(DIR, "xnor_kernel.so"))
blas_lib = ctypes.CDLL(os.path.join(DIR, "fp32_blas_kernel.so"))


# ---- ctypes signatures ----------------------------------------------------
xnor_lib.cpu_set_num_threads.argtypes = [c_int]
xnor_lib.cpu_set_num_threads.restype  = None

xnor_lib.xnor_has_avx512.argtypes = []
xnor_lib.xnor_has_avx512.restype  = c_int
xnor_lib.xnor_set_avx512.argtypes = [c_int]
xnor_lib.xnor_set_avx512.restype  = None

xnor_lib.xnor_gemm.argtypes = [
    POINTER(c_uint64), POINTER(c_uint64), POINTER(c_int32),
    c_int, c_int, c_int, c_int,
]
xnor_lib.xnor_gemm.restype = None
xnor_lib.xnor_gemm_avx512.argtypes = xnor_lib.xnor_gemm.argtypes
xnor_lib.xnor_gemm_avx512.restype  = None

xnor_lib.pack_sign_to_bits.argtypes = [POINTER(c_float), POINTER(c_uint64), c_int, c_int]
xnor_lib.pack_sign_to_bits.restype  = None

xnor_lib.binary_conv2d.argtypes = [
    POINTER(c_float), POINTER(c_uint64), POINTER(c_float),
    c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int,  # K_bits, C_in, H, W, C_out, K, stride, padding
    POINTER(c_float),
]
xnor_lib.binary_conv2d.restype = None

blas_lib.fp32_conv2d_blas.argtypes = [
    POINTER(c_float), POINTER(c_float), POINTER(c_float),
    c_int, c_int, c_int, c_int, c_int, c_int, c_int,
    POINTER(c_float),
]
blas_lib.fp32_conv2d_blas.restype = None


# Pin xnor kernel to 1 thread (already default, be explicit).
xnor_lib.cpu_set_num_threads(1)
assert xnor_lib.xnor_has_avx512() == 1, "Host or build lacks AVX-512 VPOPCNTDQ"


# ---- helpers --------------------------------------------------------------
def _fp(a):    return a.ctypes.data_as(POINTER(c_float))
def _u64p(a):  return a.ctypes.data_as(POINTER(c_uint64))
def _i32p(a):  return a.ctypes.data_as(POINTER(c_int32))


def _time_min_ms(fn, warmup=3, repeats=30):
    for _ in range(warmup): fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts) * 1000.0


def pack_signed(x_signed_2d):
    """x_signed_2d: (M, K) float32 in {-1, +1}.  Returns (M, K_words) uint64."""
    M, K = x_signed_2d.shape
    K_words = (K + 63) // 64
    out = np.zeros((M, K_words), dtype=np.uint64)
    xnor_lib.pack_sign_to_bits(_fp(np.ascontiguousarray(x_signed_2d)), _u64p(out), M, K)
    return out


# ---- ResNet18 binary-path layer shapes ------------------------------------
LAYERS = [
    # (name, C_in, H, W, C_out, K, stride, padding)
    ("layer1 64->64",       64,  32, 32,  64, 3, 1, 1),
    ("layer2 64->128/s2",   64,  32, 32, 128, 3, 2, 1),
    ("layer2 128->128",    128,  16, 16, 128, 3, 1, 1),
    ("layer3 128->256/s2", 128,  16, 16, 256, 3, 2, 1),
    ("layer3 256->256",    256,   8,  8, 256, 3, 1, 1),
    ("layer4 256->512/s2", 256,   8,  8, 512, 3, 2, 1),
    ("layer4 512->512",    512,   4,  4, 512, 3, 1, 1),
]


# ===========================================================================
# Section A — GEMM micro-bench (pure XNOR kernel, no im2col)
# ===========================================================================
print("=" * 96)
print("Section A — XNOR-GEMM micro-bench (scalar popcnt vs AVX-512 VPOPCNTDQ)")
print("=" * 96)
print(f"{'shape (M×N×K_bits)':<30} {'scalar (ms)':>12} {'AVX-512 (ms)':>13} {'speedup':>9}   bit-exact?")
print("-" * 96)

gemm_total_scalar = 0.0
gemm_total_avx    = 0.0
for name, C_in, H, W, C_out, K, stride, padding in LAYERS:
    H_out = (H + 2*padding - K) // stride + 1
    W_out = (W + 2*padding - K) // stride + 1
    M       = H_out * W_out
    N       = C_out
    K_bits  = C_in * K * K
    K_words = (K_bits + 63) // 64

    rng = np.random.default_rng(42)
    A_signed = rng.choice([-1.0, 1.0], size=(M, K_bits)).astype(np.float32)
    B_signed = rng.choice([-1.0, 1.0], size=(N, K_bits)).astype(np.float32)
    A_packed = pack_signed(A_signed)
    B_packed = pack_signed(B_signed)

    out_scalar = np.zeros((M, N), dtype=np.int32)
    out_avx512 = np.zeros((M, N), dtype=np.int32)
    xnor_lib.xnor_gemm       (_u64p(A_packed), _u64p(B_packed), _i32p(out_scalar), M, N, K_words, K_bits)
    xnor_lib.xnor_gemm_avx512(_u64p(A_packed), _u64p(B_packed), _i32p(out_avx512), M, N, K_words, K_bits)
    bit_exact = np.array_equal(out_scalar, out_avx512)
    assert bit_exact, f"BIT-EXACT FAIL on {name}: max diff = {np.abs(out_scalar - out_avx512).max()}"

    t_s = _time_min_ms(lambda: xnor_lib.xnor_gemm(
        _u64p(A_packed), _u64p(B_packed), _i32p(out_scalar), M, N, K_words, K_bits))
    t_a = _time_min_ms(lambda: xnor_lib.xnor_gemm_avx512(
        _u64p(A_packed), _u64p(B_packed), _i32p(out_avx512), M, N, K_words, K_bits))
    gemm_total_scalar += t_s
    gemm_total_avx    += t_a
    shape_str = f"({M}, {N}, {K_bits})"
    print(f"  {name:<22} {shape_str:<18} {t_s:>10.3f}   {t_a:>11.3f}   {t_s/max(t_a,1e-6):>6.2f}×   "
          f"{'✓' if bit_exact else '✗'}")

print("-" * 96)
print(f"  {'TOTAL':<41} {gemm_total_scalar:>10.3f}   {gemm_total_avx:>11.3f}   "
      f"{gemm_total_scalar/max(gemm_total_avx,1e-6):>6.2f}×")


# ===========================================================================
# Section B — Full conv2d (FP32 BLAS vs binary scalar vs binary AVX-512)
# ===========================================================================
print()
print("=" * 96)
print("Section B — Full conv2d (im2col + GEMM + correction included)")
print("=" * 96)
print(f"{'layer':<24} {'FP32 BLAS':>11} {'XNOR scalar':>13} {'XNOR AVX-512':>14}  "
      f"{'sc/blas':>8} {'avx/blas':>9} {'avx/sc':>8}   exact?")
print("-" * 96)

blas_total = scalar_total = avx_total = 0.0
for name, C_in, H, W, C_out, K, stride, padding in LAYERS:
    H_out = (H + 2*padding - K) // stride + 1
    W_out = (W + 2*padding - K) // stride + 1
    K_bits = C_in * K * K
    K_words = (K_bits + 63) // 64

    rng = np.random.default_rng(42)
    # FP32 side: random weight/input/bias, normalized.
    x_fp = rng.standard_normal((C_in, H, W)).astype(np.float32)
    w_fp = (rng.standard_normal((C_out, C_in, K, K)).astype(np.float32) * 0.1)
    b_fp = (rng.standard_normal((C_out,)).astype(np.float32) * 0.01)
    out_blas = np.zeros((C_out, H_out, W_out), dtype=np.float32)

    # Binary side: signed input/weights ({-1,+1}). alpha per channel = 1.
    x_signed = rng.choice([-1.0, 1.0], size=(C_in, H, W)).astype(np.float32)
    w_signed = rng.choice([-1.0, 1.0], size=(C_out, C_in, K, K)).astype(np.float32)
    # Pack weight to (C_out, K_words). pack_sign_to_bits expects (M, K) flat.
    w_packed = pack_signed(w_signed.reshape(C_out, -1))
    alpha    = np.ones((C_out,), dtype=np.float32)
    out_scalar = np.zeros((C_out, H_out, W_out), dtype=np.float32)
    out_avx512 = np.zeros((C_out, H_out, W_out), dtype=np.float32)

    def call_blas():
        blas_lib.fp32_conv2d_blas(_fp(x_fp), _fp(w_fp), _fp(b_fp),
            C_in, H, W, C_out, K, stride, padding, _fp(out_blas))

    def call_bin_scalar():
        xnor_lib.xnor_set_avx512(0)
        xnor_lib.binary_conv2d(_fp(x_signed), _u64p(w_packed), _fp(alpha),
            K_bits, C_in, H, W, C_out, K, stride, padding, _fp(out_scalar))

    def call_bin_avx512():
        xnor_lib.xnor_set_avx512(1)
        xnor_lib.binary_conv2d(_fp(x_signed), _u64p(w_packed), _fp(alpha),
            K_bits, C_in, H, W, C_out, K, stride, padding, _fp(out_avx512))

    # Bit-exact check between scalar and AVX-512 binary paths.
    call_bin_scalar(); call_bin_avx512()
    bin_exact = np.array_equal(out_scalar, out_avx512)
    assert bin_exact, (f"BINARY BIT-EXACT FAIL on {name}: max diff = "
                       f"{np.abs(out_scalar - out_avx512).max()}")

    t_blas   = _time_min_ms(call_blas)
    t_scalar = _time_min_ms(call_bin_scalar)
    t_avx    = _time_min_ms(call_bin_avx512)
    blas_total   += t_blas
    scalar_total += t_scalar
    avx_total    += t_avx
    print(f"  {name:<22} {t_blas:>9.3f}   {t_scalar:>11.3f}   {t_avx:>12.3f}   "
          f"{t_scalar/max(t_blas,1e-6):>6.2f}× {t_avx/max(t_blas,1e-6):>7.2f}× "
          f"{t_scalar/max(t_avx,1e-6):>6.2f}×   {'✓' if bin_exact else '✗'}")

print("-" * 96)
print(f"  {'TOTAL':<22} {blas_total:>9.3f}   {scalar_total:>11.3f}   {avx_total:>12.3f}   "
      f"{scalar_total/max(blas_total,1e-6):>6.2f}× {avx_total/max(blas_total,1e-6):>7.2f}× "
      f"{scalar_total/max(avx_total,1e-6):>6.2f}×")
print()
print("Legend: sc/blas = XNOR-scalar  vs FP32-BLAS  ratio (<1 means binary faster)")
print("        avx/blas = XNOR-AVX512 vs FP32-BLAS  ratio")
print("        avx/sc   = AVX-512 speedup over scalar XNOR  (the VPOPCNTDQ win)")
