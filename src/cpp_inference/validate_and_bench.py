"""
Validate + benchmark xnor_kernel.so against PyTorch reference.

Run after `bash build.sh`.

Usage in Kaggle / notebook:
    import sys; sys.path.insert(0, '.agents/avx_demo')
    from validate_and_bench import validate, benchmark
    validate()
    benchmark(M=64, N=512, K=4608)        # layer4.1.conv2 GEMM shape
"""

import ctypes
import os
import time
import numpy as np
import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB = ctypes.CDLL(os.path.join(_DIR, 'xnor_kernel.so'))

# ---- ctypes prototypes ------------------------------------------------
u64p = ctypes.POINTER(ctypes.c_uint64)
i32p = ctypes.POINTER(ctypes.c_int32)
f32p = ctypes.POINTER(ctypes.c_float)

_LIB.xnor_gemm.argtypes        = [u64p, u64p, i32p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_LIB.xnor_gemm_padded.argtypes = [u64p, u64p, i32p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_LIB.pack_sign_to_bits.argtypes = [f32p, u64p, ctypes.c_int, ctypes.c_int]
_LIB.fp32_gemm_ref.argtypes     = [f32p, f32p, f32p, ctypes.c_int, ctypes.c_int, ctypes.c_int]


# ---- numpy helpers ----------------------------------------------------
def _ptr(a, ctype):
    """Return a ctypes pointer to the first element of a contiguous numpy array."""
    assert a.flags['C_CONTIGUOUS']
    return a.ctypes.data_as(ctypes.POINTER(ctype))


def pack_sign(X):
    """Pack float row-major matrix (M, K) into uint64 bits (M, K_words)."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    M, K = X.shape
    K_words = (K + 63) // 64
    Xb = np.zeros((M, K_words), dtype=np.uint64)
    _LIB.pack_sign_to_bits(_ptr(X, ctypes.c_float), _ptr(Xb, ctypes.c_uint64), M, K)
    return Xb, K


def xnor_gemm_py(Ab, Bb, K_bits, use_padded=None):
    """Compute C = sign-dot(A, B.T) where A, B are pre-packed binary.
    Ab: (M, K_words) uint64, Bb: (N, K_words) uint64. Returns int32 (M, N)."""
    M, K_words_a = Ab.shape
    N, K_words_b = Bb.shape
    assert K_words_a == K_words_b
    K_words = K_words_a
    if use_padded is None:
        use_padded = (K_bits != K_words * 64)
    C = np.zeros((M, N), dtype=np.int32)
    fn = _LIB.xnor_gemm_padded if use_padded else _LIB.xnor_gemm
    fn(_ptr(Ab, ctypes.c_uint64), _ptr(Bb, ctypes.c_uint64), _ptr(C, ctypes.c_int32),
       M, N, K_words, K_bits)
    return C


def fp32_gemm_ref(A, B):
    """Single-thread C-loop FP32 GEMM (A @ B.T). For fair head-to-head."""
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    M, K = A.shape; N, K2 = B.shape; assert K == K2
    C = np.zeros((M, N), dtype=np.float32)
    _LIB.fp32_gemm_ref(_ptr(A, ctypes.c_float), _ptr(B, ctypes.c_float),
                       _ptr(C, ctypes.c_float), M, N, K)
    return C


# ---- validation -------------------------------------------------------
def validate(M=8, N=16, K=320):
    """Compare xnor_gemm output against PyTorch sign-dot reference."""
    print(f"-- validate M={M} N={N} K={K} --")
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((N, K)).astype(np.float32)

    # Reference: sign({-1,+1}) dot product in PyTorch.
    # Our convention: x>=0 -> +1, else -1.
    A_signed = (A >= 0).astype(np.float32) * 2 - 1
    B_signed = (B >= 0).astype(np.float32) * 2 - 1
    ref = A_signed @ B_signed.T  # (M, N) float, but integer-valued

    # Ours via packed kernel.
    Ab, _ = pack_sign(A)
    Bb, _ = pack_sign(B)
    out = xnor_gemm_py(Ab, Bb, K_bits=K)

    diff = np.abs(out.astype(np.float32) - ref).max()
    print(f"   max |xnor_gemm - ref|: {diff}")
    assert diff < 0.5, "xnor_gemm output mismatches sign-dot reference"
    print("   ok ✓")


# ---- benchmark --------------------------------------------------------
def _time(fn, n=10):
    """Median of n timed runs (after 1 warmup)."""
    fn()  # warmup
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts)//2], min(ts), max(ts)


def benchmark(M=64, N=512, K=4608, n_runs=20):
    """Benchmark binary vs FP32 GEMM at one layer shape.

    M=64, N=512, K=4608 corresponds to layer4.1.conv2 of CIFAR ResNet18
    (input 4×4×512 unfolded with 3×3 kernel -> 4*4=16*4=64 spatial rows
    per image; depends on batching).
    """
    print(f"-- benchmark M={M} N={N} K={K} --")
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((N, K)).astype(np.float32)
    Ab, _ = pack_sign(A); Bb, _ = pack_sign(B)

    # Pin single-thread for fair comparison.
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS']      = '1'
    os.environ['MKL_NUM_THREADS']      = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    A_t = torch.from_numpy(A); B_t = torch.from_numpy(B)
    pack_med, _, _ = _time(lambda: pack_sign(A), n=n_runs)
    bin_med,  _, _ = _time(lambda: xnor_gemm_py(Ab, Bb, K_bits=K), n=n_runs)
    fp_ref_med, _, _ = _time(lambda: fp32_gemm_ref(A, B),         n=max(3, n_runs // 4))
    fp_torch_med, _, _ = _time(lambda: torch.matmul(A_t, B_t.T),  n=n_runs)
    # numpy multi-thread BLAS as upper-bound baseline.
    fp_np_med, _, _ = _time(lambda: A @ B.T, n=n_runs)

    print(f"   binary GEMM (popcount)        : {bin_med*1000:.3f} ms")
    print(f"   FP32 GEMM (single-thread C)   : {fp_ref_med*1000:.3f} ms   "
          f"=> speedup vs single-thread: {fp_ref_med/bin_med:.2f}x")
    print(f"   FP32 GEMM (torch, 1 thread)   : {fp_torch_med*1000:.3f} ms   "
          f"=> speedup vs torch-1t       : {fp_torch_med/bin_med:.2f}x")
    print(f"   FP32 GEMM (numpy, multi-thread): {fp_np_med*1000:.3f} ms   "
          f"=> speedup vs numpy-BLAS    : {fp_np_med/bin_med:.2f}x")
    print(f"   pack_sign(A) [excluded above] : {pack_med*1000:.3f} ms")
    return {
        'binary_ms':         bin_med * 1000,
        'fp32_singlethread': fp_ref_med * 1000,
        'fp32_torch_1t':     fp_torch_med * 1000,
        'fp32_numpy_blas':   fp_np_med * 1000,
        'pack_ms':           pack_med * 1000,
    }


if __name__ == '__main__':
    validate(M=8,  N=16,  K=320)
    validate(M=64, N=128, K=1152)   # 128x3x3
    print()
    benchmark(M=64, N=512, K=4608)  # ResNet18 layer4.1.conv2 area
    print()
    benchmark(M=256, N=256, K=2304) # ResNet18 layer3.1.conv2 area
    print()
    benchmark(M=1024, N=1024, K=4096)  # large GEMM
