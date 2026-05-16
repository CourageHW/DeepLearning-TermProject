/* ============================================================
 *  fp32_blas_kernel.c
 *
 *  BLAS-backed FP32 conv2d for fair teacher latency comparison.
 *  Same signature as `fp32_conv2d` in xnor_kernel.c, but uses
 *  im2col + cblas_sgemm under the hood.
 *
 *  Linked at build time against -lblas (reference BLAS on Debian/Ubuntu).
 *  At runtime, swap in OpenBLAS / MKL by LD_PRELOAD if desired:
 *      LD_PRELOAD=/path/to/libopenblas.so python3 inference_c.py --blas ...
 *
 *  We forward-declare cblas_sgemm to avoid needing libblas-dev / cblas.h.
 *  The ABI is fixed across all BLAS implementations.
 * ============================================================ */
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* CBLAS enums — these integer values are part of the cblas ABI. */
enum CBLAS_ORDER     { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans  = 111, CblasTrans    = 112, CblasConjTrans = 113 };

extern void cblas_sgemm(
    enum CBLAS_ORDER     Order,
    enum CBLAS_TRANSPOSE TransA,
    enum CBLAS_TRANSPOSE TransB,
    int M, int N, int K,
    float alpha,
    const float *A, int lda,
    const float *B, int ldb,
    float beta,
    float *C, int ldc
);

/* ------------------------------------------------------------
 * im2col helper.
 *   input  : (C_in, H, W)
 *   output : (C_in * K * K,  H_out * W_out)   row-major
 *
 * Each column = one K×K×C_in receptive field for one output (ho, wo).
 * Out-of-bounds positions (padding) are zero.
 * ------------------------------------------------------------ */
static void im2col_chw(
    const float *input,
    int C_in, int H, int W, int K, int stride, int padding,
    int H_out, int W_out,
    float *col   /* (C_in*K*K) x (H_out*W_out) */
) {
    int row_total = C_in * K * K;
    int col_total = H_out * W_out;
    #pragma omp parallel for schedule(static) if(col_total > 64)
    for (int oc = 0; oc < col_total; ++oc) {
        int ho = oc / W_out;
        int wo = oc % W_out;
        for (int ci = 0; ci < C_in; ++ci) {
            for (int kh = 0; kh < K; ++kh) {
                int ih = ho * stride - padding + kh;
                for (int kw = 0; kw < K; ++kw) {
                    int iw = wo * stride - padding + kw;
                    int row = (ci * K + kh) * K + kw;
                    float v = 0.0f;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        v = input[(ci * H + ih) * W + iw];
                    col[(size_t)row * col_total + oc] = v;
                }
            }
        }
    }
    (void)row_total;
}


/* ------------------------------------------------------------
 *  fp32_conv2d_blas — same signature as fp32_conv2d, BLAS-backed.
 *
 *  Algorithm:
 *    1. im2col(input)         → I  of shape (C_in*K*K) × (H_out*W_out)
 *    2. weight viewed as W of shape C_out × (C_in*K*K)
 *    3. SGEMM:  out_2d (C_out × H_out*W_out) = W · I
 *    4. Bias broadcast on row dim (output channel).
 *
 *  Output is written in NCHW (C_out, H_out, W_out) — same as
 *  fp32_conv2d in xnor_kernel.c, so the engine can swap calls 1-to-1.
 * ------------------------------------------------------------ */
void fp32_conv2d_blas(
    const float *input, const float *weight, const float *bias,
    int C_in, int H, int W, int C_out, int K, int stride, int padding,
    float *output
) {
    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;
    int K_total = C_in * K * K;         /* GEMM inner dim */
    int N       = H_out * W_out;        /* GEMM N         */

    /* Special-case 1×1 conv with stride=1, padding=0 — skip im2col. */
    int is_pointwise_dense = (K == 1 && stride == 1 && padding == 0);

    float *col = NULL;
    const float *gemm_B;
    if (is_pointwise_dense) {
        /* input is already (C_in, H_out*W_out) when reshaped. */
        gemm_B = input;
    } else {
        col = (float *)aligned_alloc(64,
                ((size_t)K_total * N * sizeof(float) + 63) & ~(size_t)63);
        im2col_chw(input, C_in, H, W, K, stride, padding, H_out, W_out, col);
        gemm_B = col;
    }

    /* C (C_out × N)  =  1.0 * W (C_out × K_total) * gemm_B (K_total × N)  +  0 * C */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                C_out, N, K_total,
                1.0f, weight, K_total,
                      gemm_B, N,
                0.0f, output, N);

    /* Bias add: output[co, :] += bias[co]. */
    if (bias != NULL) {
        #pragma omp parallel for schedule(static) if(C_out > 8)
        for (int co = 0; co < C_out; ++co) {
            float b = bias[co];
            float *row = output + (size_t)co * N;
            for (int i = 0; i < N; ++i) row[i] += b;
        }
    }

    if (col) free(col);
}
