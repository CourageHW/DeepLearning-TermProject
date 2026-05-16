#!/bin/bash
# Compile xnor_kernel.c -> xnor_kernel.so
# Usage: bash build.sh
#
# Flags rationale:
#   -O3           full optimisation
#   -march=native on AVX-512 hosts this enables AVX-512F/BW/DQ/VPOPCNTDQ,
#                 which exposes __AVX512VPOPCNTDQ__ in the TU and unlocks
#                 the explicit-intrinsic xnor_gemm_avx512 path.
#                 The scalar xnor_gemm/xnor_gemm_padded are wrapped in a
#                 `#pragma GCC target("arch=haswell")` block so they stay
#                 scalar-popcnt regardless of host (clean A/B baseline).
#   -mpopcnt      explicit scalar POPCNT emission for __builtin_popcountll
#   -mavx2        baseline SIMD
#   -fopenmp      OpenMP. Runtime selects 1..N threads via
#                 `cpu_set_num_threads(n)` (default 1 keeps single-thread parity).
#   -fPIC -shared shared object for ctypes loading
set -e
DIR="$(dirname "$0")"

# --- Main XNOR / FP32-naive kernel ---
gcc -O3 -march=native -mpopcnt -mavx2 -fopenmp -fPIC -shared \
    -o "$DIR/xnor_kernel.so" "$DIR/xnor_kernel.c"
echo "built $DIR/xnor_kernel.so"

# --- Sanity: AVX-512 path compiled in? scalar path still scalar? ---
if nm -D "$DIR/xnor_kernel.so" 2>/dev/null | grep -q ' T xnor_gemm_avx512$'; then
    echo "  ✓ xnor_gemm_avx512 symbol present (AVX-512 VPOPCNTDQ path active)"
else
    echo "  ✗ xnor_gemm_avx512 symbol MISSING (host lacks AVX-512 VPOPCNTDQ?)"
fi
# Inspect the OpenMP work functions (`<fn>._omp_fn.0`) — that's where the inner
# loop lives. `set +e` because grep -c returns 1 when count is 0.
set +e
inspect_fn() {
    local fn="$1"
    local body
    body=$(objdump -d "$DIR/xnor_kernel.so" 2>/dev/null \
        | awk -v t="<${fn}._omp_fn.0>:" '$0 ~ t {f=1; next} /^[0-9a-f]+ <[a-zA-Z]/{if(f){exit}} f')
    local npop nvpop
    npop=$(echo "$body"  | grep -cE '\bpopcnt\b')
    nvpop=$(echo "$body" | grep -cE '\bvpopcntq\b')
    printf "  %-22s popcnt=%-3d vpopcntq=%-3d" "$fn" "$npop" "$nvpop"
    if [ "$fn" = "xnor_gemm" ] && [ "$nvpop" -gt 0 ]; then
        echo "  ⚠ scalar path leaked vpopcntq — A/B baseline contaminated"
    elif [ "$fn" = "xnor_gemm_avx512" ] && [ "$nvpop" -eq 0 ]; then
        echo "  ⚠ AVX-512 path has no vpopcntq — kernel didn't compile correctly"
    else
        echo ""
    fi
}
inspect_fn xnor_gemm
inspect_fn xnor_gemm_avx512
set -e

# --- Optional BLAS-backed FP32 conv (for fair teacher benchmark) ---
# Links against -lblas (Debian/Ubuntu reference BLAS). At runtime, swap in
# OpenBLAS / MKL via LD_PRELOAD if installed:
#   LD_PRELOAD=.../libopenblas.so python3 inference_c.py --blas ...
BLAS_PATH=$(ldconfig -p 2>/dev/null | awk '/libblas\.so\.3 /{print $NF; exit}')
if [ -n "$BLAS_PATH" ]; then
    gcc -O3 -march=native -fopenmp -fPIC -shared \
        -o "$DIR/fp32_blas_kernel.so" "$DIR/fp32_blas_kernel.c" -l:libblas.so.3
    echo "built $DIR/fp32_blas_kernel.so (linked: $BLAS_PATH)"
else
    echo "SKIPPED $DIR/fp32_blas_kernel.so — libblas.so.3 not found"
fi

echo "CPU features:"
grep -m1 -o '\(avx2\|popcnt\|avx512f\|avx512vpopcntdq\)' /proc/cpuinfo | sort -u
echo "logical cores: $(nproc)"
