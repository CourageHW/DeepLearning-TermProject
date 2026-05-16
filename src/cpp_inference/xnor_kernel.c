/*
 * xnor_kernel.c — Binary inference primitives for A1W1ResNet18v2
 *
 * Build: see build.sh.  Single-thread, AVX2 + POPCNT baseline.
 *
 * Public API (all functions are `extern "C"`-equivalent for ctypes use):
 *   xnor_gemm                Binary matrix multiply (sign-dot via popcount)
 *   xnor_gemm_padded         Same, with mask for trailing pad bits
 *   pack_sign_to_bits        float matrix -> packed binary bits
 *   fp32_gemm_ref            Naive FP32 GEMM (reference)
 *   fp32_conv2d              FP32 convolution (used for stem)
 *   batch_norm_2d            Inference-time batch normalization
 *   rsign                    Sign with per-channel learnable threshold
 *   rprelu                   PReLU(x - gamma) + beta
 *   binary_conv2d            Full binary 2D convolution with padding correction
 *   shortcut_downsample      AvgPool + zero-pad-channels
 *   adaptive_avgpool_1       Global average pooling to (1,1)
 *   linear_fp32              FP32 linear layer
 *   add_inplace              elementwise add
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* --- Runtime thread control ------------------------------------------
 * Default to 1 thread (single-core behaviour preserved unless caller sets it).
 * Call cpu_set_num_threads(N) from Python (ctypes) before forward() to enable
 * parallel execution. Each parallel kernel guards with `if (omp_get_max_threads() > 1)`
 * + a workload threshold so tiny layers don't pay thread-spawn overhead. */
static __attribute__((constructor)) void _init_threads(void) { omp_set_num_threads(1); }
void cpu_set_num_threads(int n) { if (n < 1) n = 1; omp_set_num_threads(n); }
int  cpu_get_max_threads(void)   { return omp_get_max_threads(); }


/* ============================================================
 * Section 1: Binary GEMM primitives (scalar popcount baseline)
 *
 * Wrapped in `#pragma GCC target("arch=haswell")` to compile these two
 * functions as if -march=haswell were in effect (AVX2 + scalar POPCNT,
 * NO AVX-512). This guarantees scalar `popcnt r64` instructions in the
 * inner loop even when the rest of the TU is built with -march=native
 * on an AVX-512 capable host. The AVX-512 VPOPCNTDQ variant lives in
 * Section 1.5 below and is what the bar chart compares against.
 * ============================================================ */
#pragma GCC push_options
#pragma GCC target("arch=haswell")

void xnor_gemm(
    const uint64_t *A, const uint64_t *B, int32_t *C,
    int M, int N, int K_words, int K_bits
) {
    const int N_BLK = 16;
    #pragma omp parallel for schedule(static) if(M > 32)
    for (int i = 0; i < M; ++i) {
        const uint64_t *a = A + (size_t)i * K_words;
        int j = 0;
        for (; j + N_BLK <= N; j += N_BLK) {
            int32_t acc[N_BLK];
            for (int jb = 0; jb < N_BLK; ++jb) acc[jb] = 0;
            for (int k = 0; k < K_words; ++k) {
                uint64_t av = a[k];
                for (int jb = 0; jb < N_BLK; ++jb) {
                    uint64_t bv = B[(size_t)(j + jb) * K_words + k];
                    acc[jb] += __builtin_popcountll(~(av ^ bv));
                }
            }
            for (int jb = 0; jb < N_BLK; ++jb)
                C[(size_t)i * N + j + jb] = 2 * acc[jb] - K_bits;
        }
        for (; j < N; ++j) {
            const uint64_t *b = B + (size_t)j * K_words;
            int32_t pc = 0;
            for (int k = 0; k < K_words; ++k)
                pc += __builtin_popcountll(~(a[k] ^ b[k]));
            C[(size_t)i * N + j] = 2 * pc - K_bits;
        }
    }
}

void xnor_gemm_padded(
    const uint64_t *A, const uint64_t *B, int32_t *C,
    int M, int N, int K_words, int K_bits
) {
    int pad_bits = K_words * 64 - K_bits;
    for (int i = 0; i < M; ++i) {
        const uint64_t *a = A + (size_t)i * K_words;
        for (int j = 0; j < N; ++j) {
            const uint64_t *b = B + (size_t)j * K_words;
            int32_t pc = 0;
            for (int k = 0; k < K_words; ++k)
                pc += __builtin_popcountll(~(a[k] ^ b[k]));
            pc -= pad_bits;
            C[(size_t)i * N + j] = 2 * pc - K_bits;
        }
    }
}

#pragma GCC pop_options


/* ============================================================
 * Section 1.5: AVX-512 VPOPCNTDQ accelerated binary GEMM
 *
 * Identical semantics to xnor_gemm. Same memory layout, same output
 * formula `2 * popcount(XNOR(A, B)) - K_bits`. Uses `_mm512_popcnt_epi64`
 * (AVX512_VPOPCNTDQ) to popcount 8x uint64 per cycle.
 *
 * Implementation tracks popcount(XOR) instead of popcount(XNOR); they
 * differ only by `K_words*64 - xor_pop = xnor_pop`. This saves one vector
 * NOT per inner iteration and is algebraically equivalent.
 *
 * Tail handling: K_words may not be a multiple of 8 (e.g., 9, 18, 36).
 * The vectorized loop processes `(K_words / 8) * 8` words, then a
 * scalar __builtin_popcountll loop handles the remainder.
 *
 * Compile-time gated on __AVX512VPOPCNTDQ__ so the same source builds
 * on older hosts (then xnor_has_avx512() returns 0).
 * ============================================================ */
#if defined(__AVX512VPOPCNTDQ__) && defined(__AVX512F__) && defined(__AVX512BW__)
#define HAVE_AVX512_VPOPCNTDQ 1
#include <immintrin.h>

void xnor_gemm_avx512(
    const uint64_t *A, const uint64_t *B, int32_t *C,
    int M, int N, int K_words, int K_bits
) {
    const int N_BLK = 4;            /* 4 B-rows per inner iteration */
    const int K_VEC = 8;            /* uint64 per __m512i */
    const int K_main = (K_words / K_VEC) * K_VEC;
    const int total_bits = K_words * 64;

    #pragma omp parallel for schedule(static) if(M > 32)
    for (int i = 0; i < M; ++i) {
        const uint64_t *a = A + (size_t)i * K_words;
        int j = 0;
        for (; j + N_BLK <= N; j += N_BLK) {
            const uint64_t *b0 = B + (size_t)(j + 0) * K_words;
            const uint64_t *b1 = B + (size_t)(j + 1) * K_words;
            const uint64_t *b2 = B + (size_t)(j + 2) * K_words;
            const uint64_t *b3 = B + (size_t)(j + 3) * K_words;
            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512();
            __m512i acc3 = _mm512_setzero_si512();
            for (int k = 0; k < K_main; k += K_VEC) {
                __m512i av  = _mm512_loadu_si512((const __m512i *)(a  + k));
                __m512i bv0 = _mm512_loadu_si512((const __m512i *)(b0 + k));
                __m512i bv1 = _mm512_loadu_si512((const __m512i *)(b1 + k));
                __m512i bv2 = _mm512_loadu_si512((const __m512i *)(b2 + k));
                __m512i bv3 = _mm512_loadu_si512((const __m512i *)(b3 + k));
                acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(_mm512_xor_si512(av, bv0)));
                acc1 = _mm512_add_epi64(acc1, _mm512_popcnt_epi64(_mm512_xor_si512(av, bv1)));
                acc2 = _mm512_add_epi64(acc2, _mm512_popcnt_epi64(_mm512_xor_si512(av, bv2)));
                acc3 = _mm512_add_epi64(acc3, _mm512_popcnt_epi64(_mm512_xor_si512(av, bv3)));
            }
            int64_t x0 = _mm512_reduce_add_epi64(acc0);
            int64_t x1 = _mm512_reduce_add_epi64(acc1);
            int64_t x2 = _mm512_reduce_add_epi64(acc2);
            int64_t x3 = _mm512_reduce_add_epi64(acc3);
            for (int k = K_main; k < K_words; ++k) {
                x0 += __builtin_popcountll(a[k] ^ b0[k]);
                x1 += __builtin_popcountll(a[k] ^ b1[k]);
                x2 += __builtin_popcountll(a[k] ^ b2[k]);
                x3 += __builtin_popcountll(a[k] ^ b3[k]);
            }
            C[(size_t)i * N + j + 0] = 2 * (int32_t)(total_bits - x0) - K_bits;
            C[(size_t)i * N + j + 1] = 2 * (int32_t)(total_bits - x1) - K_bits;
            C[(size_t)i * N + j + 2] = 2 * (int32_t)(total_bits - x2) - K_bits;
            C[(size_t)i * N + j + 3] = 2 * (int32_t)(total_bits - x3) - K_bits;
        }
        for (; j < N; ++j) {
            const uint64_t *b = B + (size_t)j * K_words;
            __m512i acc = _mm512_setzero_si512();
            for (int k = 0; k < K_main; k += K_VEC) {
                __m512i av = _mm512_loadu_si512((const __m512i *)(a + k));
                __m512i bv = _mm512_loadu_si512((const __m512i *)(b + k));
                acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(_mm512_xor_si512(av, bv)));
            }
            int64_t x = _mm512_reduce_add_epi64(acc);
            for (int k = K_main; k < K_words; ++k)
                x += __builtin_popcountll(a[k] ^ b[k]);
            C[(size_t)i * N + j] = 2 * (int32_t)(total_bits - x) - K_bits;
        }
    }
}
#else
#define HAVE_AVX512_VPOPCNTDQ 0
#endif

/* --- Dispatch state + queries ---------------------------------------- */
static int g_xnor_use_avx512 = 0;

int  xnor_has_avx512(void)        { return HAVE_AVX512_VPOPCNTDQ; }
int  xnor_get_avx512(void)        { return g_xnor_use_avx512; }
void xnor_set_avx512(int enable)  {
    g_xnor_use_avx512 = (HAVE_AVX512_VPOPCNTDQ && enable) ? 1 : 0;
}

static inline void xnor_gemm_dispatch(
    const uint64_t *A, const uint64_t *B, int32_t *C,
    int M, int N, int K_words, int K_bits
) {
#if HAVE_AVX512_VPOPCNTDQ
    if (g_xnor_use_avx512) {
        xnor_gemm_avx512(A, B, C, M, N, K_words, K_bits);
        return;
    }
#endif
    xnor_gemm(A, B, C, M, N, K_words, K_bits);
}

void pack_sign_to_bits(const float *X, uint64_t *Xb, int M, int K) {
    int K_words = (K + 63) / 64;
    for (int i = 0; i < M; ++i) {
        for (int kw = 0; kw < K_words; ++kw) {
            uint64_t w = 0;
            int kstart = kw * 64;
            int kend = kstart + 64; if (kend > K) kend = K;
            for (int k = kstart; k < kend; ++k)
                if (X[(size_t)i * K + k] >= 0.0f)
                    w |= (uint64_t)1 << (k - kstart);
            Xb[(size_t)i * K_words + kw] = w;
        }
    }
}

void fp32_gemm_ref(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            const float *a = A + (size_t)i * K;
            const float *b = B + (size_t)j * K;
            for (int k = 0; k < K; ++k) acc += a[k] * b[k];
            C[(size_t)i * N + j] = acc;
        }
    }
}


/* ============================================================
 * Section 2: FP32 conv2d (stem)
 *
 * NCHW layout, batch=1 assumed.
 * input  : (C_in,  H,    W)
 * weight : (C_out, C_in, K, K)
 * bias   : (C_out,) or NULL
 * output : (C_out, H_out, W_out)
 * ============================================================ */
void fp32_conv2d(
    const float *input, const float *weight, const float *bias,
    int C_in, int H, int W, int C_out, int K, int stride, int padding,
    float *output
) {
    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;
    #pragma omp parallel for schedule(static) if(C_out > 8)
    for (int co = 0; co < C_out; ++co) {
        for (int ho = 0; ho < H_out; ++ho) {
            for (int wo = 0; wo < W_out; ++wo) {
                float acc = bias ? bias[co] : 0.0f;
                for (int ci = 0; ci < C_in; ++ci) {
                    for (int kh = 0; kh < K; ++kh) {
                        int ih = ho * stride - padding + kh;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < K; ++kw) {
                            int iw = wo * stride - padding + kw;
                            if (iw < 0 || iw >= W) continue;
                            acc += input[(ci * H + ih) * W + iw]
                                 * weight[((co * C_in + ci) * K + kh) * K + kw];
                        }
                    }
                }
                output[(co * H_out + ho) * W_out + wo] = acc;
            }
        }
    }
}


/* ============================================================
 * Section 3: Batch normalization (inference)
 *   y_c = (x_c - mean_c) / sqrt(var_c + eps) * weight_c + bias_c
 * ============================================================ */
void batch_norm_2d(
    const float *input,
    const float *mean, const float *var, const float *weight, const float *bias,
    int C, int H, int W, float eps,
    float *output
) {
    int HW = H * W;
    #pragma omp parallel for schedule(static) if(C > 32)
    for (int c = 0; c < C; ++c) {
        float inv = 1.0f / sqrtf(var[c] + eps);
        float a = weight[c] * inv;
        float b = bias[c] - mean[c] * a;
        const float *xc = input + c * HW;
        float *yc = output + c * HW;
        for (int i = 0; i < HW; ++i) yc[i] = a * xc[i] + b;
    }
}


/* ============================================================
 * Section 4: RSign (per-channel learnable threshold)
 *   y_c = sign(x_c - threshold_c)   { -1, +1 } as float
 * ============================================================ */
void rsign(
    const float *input, const float *threshold,
    int C, int H, int W,
    float *output
) {
    int HW = H * W;
    #pragma omp parallel for schedule(static) if(C > 32)
    for (int c = 0; c < C; ++c) {
        float t = threshold[c];
        const float *xc = input + c * HW;
        float *yc = output + c * HW;
        for (int i = 0; i < HW; ++i) yc[i] = (xc[i] - t >= 0.0f) ? 1.0f : -1.0f;
    }
}


/* ============================================================
 * Section 5: RPReLU
 *   y = PReLU(x - gamma) + beta
 *   PReLU(z, a) = max(z, 0) + a * min(z, 0)
 * ============================================================ */
void rprelu(
    const float *input,
    const float *gamma, const float *beta, const float *prelu_weight,
    int C, int H, int W,
    float *output
) {
    int HW = H * W;
    #pragma omp parallel for schedule(static) if(C > 32)
    for (int c = 0; c < C; ++c) {
        float g = gamma[c], b = beta[c], a = prelu_weight[c];
        const float *xc = input + c * HW;
        float *yc = output + c * HW;
        for (int i = 0; i < HW; ++i) {
            float z = xc[i] - g;
            yc[i] = (z >= 0.0f ? z : a * z) + b;
        }
    }
}


/* ============================================================
 * Section 6: Binary conv2d with padding correction
 *
 * Logic:
 *   1. For each output position, pack the K×K×C_in receptive field
 *      (already ±1 floats from RSign) into uint64 bits. Out-of-bounds
 *      positions get bit=0 (= -1 in our sign encoding).
 *   2. Call xnor_gemm to compute raw sign-dot products.
 *   3. Apply per-position correction: zero padding originally contributes 0,
 *      but our packed-as-(-1) representation makes it contribute -w_b.
 *      So true_output = raw_output + sum_{padded} w_b
 *
 * We precompute correction per distinct padding pattern (at most 9 for K=3),
 * keyed by which (kh, kw) positions are out-of-bounds.
 * ============================================================ */
/* Pattern id = (pad_kh_mask << K) | pad_kw_mask, where bit kh in pad_kh_mask
   is set iff kernel row kh accesses out-of-bounds input at this output row.
   Supports any (stride, padding) — not just edge-vs-interior. */
static inline int pad_pattern_id(int pad_kh_mask, int pad_kw_mask, int K) {
    return (pad_kh_mask << K) | pad_kw_mask;
}

static void compute_pad_correction_one(
    const uint64_t *weight_packed,
    int C_in, int C_out, int K, int K_words,
    int pad_kh_mask, int pad_kw_mask,
    int32_t *correction_out  /* (C_out,) */
) {
    int K_bits_total = C_in * K * K;
    uint64_t *mask = (uint64_t *)calloc(K_words, sizeof(uint64_t));
    int padded_count = 0;
    for (int ci = 0; ci < C_in; ++ci) {
        for (int kh = 0; kh < K; ++kh) {
            int kh_pad = (pad_kh_mask >> kh) & 1;
            for (int kw = 0; kw < K; ++kw) {
                int kw_pad = (pad_kw_mask >> kw) & 1;
                if (kh_pad || kw_pad) {
                    int bit_idx = (ci * K + kh) * K + kw;
                    mask[bit_idx / 64] |= (uint64_t)1 << (bit_idx % 64);
                    padded_count++;
                }
            }
        }
    }
    /* For each output channel, correction = sum w_b at padded positions =
       2 * popcount(weight & mask) - padded_count. */
    for (int co = 0; co < C_out; ++co) {
        int pc = 0;
        const uint64_t *wp = weight_packed + (size_t)co * K_words;
        for (int kw = 0; kw < K_words; ++kw)
            pc += __builtin_popcountll(wp[kw] & mask[kw]);
        correction_out[co] = 2 * pc - padded_count;
    }
    (void)K_bits_total;
    free(mask);
}

void binary_conv2d(
    const float *input_signed,        /* (C_in, H, W), values in {-1,+1} */
    const uint64_t *weight_packed,    /* (C_out, K_words) */
    const float *alpha,               /* (C_out,) */
    int K_bits, int C_in, int H, int W, int C_out, int K, int stride, int padding,
    float *output                     /* (C_out, H_out, W_out) */
) {
    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;
    int K_words = (K_bits + 63) / 64;
    int M = H_out * W_out;

    uint64_t *im2col_packed = (uint64_t *)malloc((size_t)M * K_words * sizeof(uint64_t));
    int32_t *raw_out = (int32_t *)malloc((size_t)M * C_out * sizeof(int32_t));

    /* --- Pack im2col, marking padded positions as bit=0 (-1) --- */
    #pragma omp parallel for schedule(static) if(M > 32)
    for (int idx = 0; idx < M; ++idx) {
        int ho = idx / W_out, wo = idx % W_out;
        uint64_t *row = im2col_packed + (size_t)idx * K_words;
        memset(row, 0, K_words * sizeof(uint64_t));
        for (int ci = 0; ci < C_in; ++ci) {
            for (int kh = 0; kh < K; ++kh) {
                int ih = ho * stride - padding + kh;
                for (int kw = 0; kw < K; ++kw) {
                    int iw = wo * stride - padding + kw;
                    int bit_idx = (ci * K + kh) * K + kw;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        float v = input_signed[(ci * H + ih) * W + iw];
                        if (v >= 0.0f)
                            row[bit_idx / 64] |= (uint64_t)1 << (bit_idx % 64);
                    }
                    /* else: padded -> leave bit 0 (= -1 in sign encoding) */
                }
            }
        }
    }

    /* --- xnor_gemm: raw sign-dot products (treating padded as -1) ---
     * Dispatches to AVX-512 VPOPCNTDQ variant when xnor_set_avx512(1) is set. */
    xnor_gemm_dispatch(im2col_packed, weight_packed, raw_out, M, C_out, K_words, K_bits);

    /* --- Pass 1 (serial): identify unique padding patterns + precompute corrections --- */
    const int MAX_PATTERNS = 1 << (2 * K);
    int32_t **corr_cache = (int32_t **)calloc(MAX_PATTERNS, sizeof(int32_t *));
    for (int ho = 0; ho < H_out; ++ho) {
        int pad_kh_mask = 0;
        for (int kh = 0; kh < K; ++kh) {
            int ih = ho * stride - padding + kh;
            if (ih < 0 || ih >= H) pad_kh_mask |= (1 << kh);
        }
        for (int wo = 0; wo < W_out; ++wo) {
            int pad_kw_mask = 0;
            for (int kw = 0; kw < K; ++kw) {
                int iw = wo * stride - padding + kw;
                if (iw < 0 || iw >= W) pad_kw_mask |= (1 << kw);
            }
            int pid = pad_pattern_id(pad_kh_mask, pad_kw_mask, K);
            if (pid != 0 && !corr_cache[pid]) {
                corr_cache[pid] = (int32_t *)malloc(C_out * sizeof(int32_t));
                compute_pad_correction_one(weight_packed, C_in, C_out, K, K_words,
                                           pad_kh_mask, pad_kw_mask, corr_cache[pid]);
            }
        }
    }

    /* --- Pass 2 (parallel): apply correction + alpha + reshape --- */
    #pragma omp parallel for schedule(static) if(M > 32)
    for (int idx = 0; idx < M; ++idx) {
        int ho = idx / W_out, wo = idx % W_out;
        int pad_kh_mask = 0, pad_kw_mask = 0;
        for (int kh = 0; kh < K; ++kh) {
            int ih = ho * stride - padding + kh;
            if (ih < 0 || ih >= H) pad_kh_mask |= (1 << kh);
        }
        for (int kw = 0; kw < K; ++kw) {
            int iw = wo * stride - padding + kw;
            if (iw < 0 || iw >= W) pad_kw_mask |= (1 << kw);
        }
        int pid = pad_pattern_id(pad_kh_mask, pad_kw_mask, K);
        int32_t *raw_row = raw_out + (size_t)idx * C_out;
        if (pid == 0) {
            for (int co = 0; co < C_out; ++co)
                output[(co * H_out + ho) * W_out + wo] = alpha[co] * raw_row[co];
        } else {
            int32_t *corr = corr_cache[pid];
            for (int co = 0; co < C_out; ++co)
                output[(co * H_out + ho) * W_out + wo] = alpha[co] * (raw_row[co] + corr[co]);
        }
    }

    for (int i = 0; i < MAX_PATTERNS; ++i) free(corr_cache[i]);
    free(corr_cache);
    free(im2col_packed); free(raw_out);
}


/* ============================================================
 * Section 7: Shortcut downsample (parameter-free)
 *   AvgPool(stride) along H,W, then zero-pad channels 0..C_in-1 with input,
 *                                   C_in..C_out-1 with zeros.
 * ============================================================ */
void shortcut_downsample(
    const float *input,
    int C_in, int H, int W, int C_out, int stride,
    float *output
) {
    int H_out = H / stride;
    int W_out = W / stride;
    int HoWo = H_out * W_out;
    if (stride == 1) {
        /* No pooling: copy input to first C_in channels, zero rest. */
        for (int c = 0; c < C_in; ++c)
            memcpy(output + c * HoWo, input + c * H * W, HoWo * sizeof(float));
    } else {
        int s2 = stride * stride;
        for (int c = 0; c < C_in; ++c) {
            for (int ho = 0; ho < H_out; ++ho) {
                for (int wo = 0; wo < W_out; ++wo) {
                    float acc = 0.0f;
                    for (int dh = 0; dh < stride; ++dh)
                        for (int dw = 0; dw < stride; ++dw)
                            acc += input[(c * H + ho * stride + dh) * W + wo * stride + dw];
                    output[(c * H_out + ho) * W_out + wo] = acc / (float)s2;
                }
            }
        }
    }
    /* Pad remaining channels with zeros. */
    for (int c = C_in; c < C_out; ++c)
        memset(output + (size_t)c * HoWo, 0, HoWo * sizeof(float));
}


/* ============================================================
 * Section 8: Adaptive avg pool to (1,1) and FP32 linear
 * ============================================================ */
void adaptive_avgpool_1(const float *input, int C, int H, int W, float *output) {
    int HW = H * W;
    float inv = 1.0f / (float)HW;
    for (int c = 0; c < C; ++c) {
        float s = 0.0f;
        const float *xc = input + c * HW;
        for (int i = 0; i < HW; ++i) s += xc[i];
        output[c] = s * inv;
    }
}

void linear_fp32(
    const float *input, const float *weight, const float *bias,
    int in_features, int out_features,
    float *output
) {
    for (int o = 0; o < out_features; ++o) {
        float s = bias ? bias[o] : 0.0f;
        const float *wr = weight + (size_t)o * in_features;
        for (int i = 0; i < in_features; ++i) s += wr[i] * input[i];
        output[o] = s;
    }
}


/* ============================================================
 * Section 9: Elementwise add (in-place) + ReLU (in-place)
 * ============================================================ */
void add_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; ++i) a[i] += b[i];
}

void relu_inplace(float *a, int n) {
    for (int i = 0; i < n; ++i) if (a[i] < 0.0f) a[i] = 0.0f;
}
