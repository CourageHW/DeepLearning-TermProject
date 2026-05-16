# Plan v7 — CUDA C 풀 구현 (RTX 3060 데모)

W1A1 ResNet18을 **순수 CUDA C 커널**로 GPU에서 추론.
Python 없이 `.cu` 컴파일 → `.so`/standalone binary.

---

## 0. 목표 & 성공 기준

| 항목 | 현재 (CPU C) | 목표 (CUDA C) |
|---|---|---|
| Hardware | x86 AVX | **RTX 3060 (Ampere SM_86)** |
| Binary 가속 | scalar POPCNT | **`__popc()` parallel + (옵션) BMMA tensor core** |
| Latency / sample | 22 ms (single thread) | **목표 < 1 ms** |
| Throughput | 45 img/s | **목표 5000-50000 img/s** |
| 의존성 | gcc + libc | **nvcc + CUDA runtime** |
| 정확도 | bit-exact vs PyTorch | **동일** |

---

## 1. 접근 방식 비교

### Option A — Standalone CUDA C binary (권장, CPU 데모와 일관)
```
cuda_demo/
├── xnor_kernel.cu        # 모든 primitive를 CUDA 커널로
├── resnet18_binary.cu    # forward 오케스트레이션 (host 코드)
├── main.cu               # CLI, 이미지 로딩
├── weights.cu            # .bnn 파서 (host 코드)
└── Makefile              # nvcc 빌드
```
실행: `./bnn_infer_cuda student.bnn testset.bin --max 1000`

**장점**: CPU C 데모와 구조 동일 → 발표에서 "C → CUDA C 포팅" 메시지 깔끔.

### Option B — PyTorch C++ extension (.cu만 작성)
```python
import torch
import bnn_cuda  # 우리 .cu로 빌드한 extension
y = bnn_cuda.binary_conv2d(x, weight_packed, alpha, ...)
```
**장점**: PyTorch에 통합. 자동 backward 가능 (학습에도 사용).
**단점**: 발표 narrative에서 "PyTorch 안에서 customer kernel" — 약간 약함.

### 권장: **Option A**, 텀프 데모에 가장 적합. (이후 학습에도 쓰려면 B로 확장.)

---

## 2. CUDA 가속 전략 — 3 레벨

### L1 — Naive CUDA (`__popc()` 사용, 권장 시작점)
- 각 thread = 1 output position
- Inner loop에서 `__popcll(xnor_word)` (scalar popcount, 1 cycle)
- 1 SM = 32 threads/warp × 32 warps = 1024 threads
- RTX 3060: 30 SMs × 1024 = ~30K 동시 threads
- **예상 speedup vs CPU**: 50-200× (메모리 대역폭 bound)
- **구현 난이도**: 중. 우리 CPU 커널 직접 포팅.

### L2 — Shared memory + tiling (정통 CUDA 최적화)
- Weight를 shared memory에 캐싱
- Activation tile load → 여러 output 공유
- 메모리 대역폭 절감 → 추가 2-3× speedup
- **난이도**: 중상.

### L3 — WMMA BMMA Tensor Cores (최대 임팩트)
- `nvcuda::wmma::experimental::precision::b1` + `bmma_sync(c, a, b, c, bmmaBitOpXOR_POPC)`
- 8×8×128 binary GEMM 한 번에
- 1 Tensor Core × 1024 binary MAC/cycle
- 30 SMs × 4 TC = 120 TC × 1024 = **~150 TOPS** (이론)
- **예상 speedup vs L1**: 10-30×
- **난이도**: 상. 메모리 layout 매우 까다로움 (NHWC, 128-bit align, im2col 재구성)

---

## 3. CUDA Primitives (mirror our CPU)

| Primitive | CPU 코드 | CUDA 변환 |
|---|---|---|
| `xnor_gemm` | scalar loop | **L1**: 1 thread = 1 output element, `__popcll()` |
| `fp32_conv2d` (stem) | naive triple loop | cuDNN convForward OR custom kernel |
| `batch_norm_2d` | per-channel loop | element-wise kernel, broadcast |
| `rsign` | per-channel | element-wise |
| `rprelu` | per-channel | element-wise |
| `binary_conv2d` | im2col + xnor_gemm | im2col kernel + binary GEMM kernel |
| `shortcut_downsample` | avgpool + pad | element-wise kernel |
| `adaptive_avgpool_1` | sum + mean | reduction kernel |
| `linear_fp32` | matmul | cuBLAS sgemv OR custom |
| `add_inplace` | element-wise | element-wise (trivial) |

각 primitive 별 .cu 파일 또는 한 파일에 모듈로.

---

## 4. 핵심 커널 — L1 구현 스케치

### `binary_conv2d_cuda` (L1)
```cuda
__global__ void binary_conv2d_kernel(
    const float    *input_signed,    // (C_in, H, W) — ±1 float
    const uint64_t *weight_packed,   // (C_out, K_words)
    const float    *alpha,           // (C_out,)
    float          *output,          // (C_out, H_out, W_out)
    int C_in, int H, int W, int C_out, int K, int stride, int padding,
    int K_bits, int K_words
) {
    int co = blockIdx.x;                  // output channel
    int ho = blockIdx.y;                  // output row
    int wo = threadIdx.x;                 // output col
    if (wo >= W_out) return;

    // Build packed im2col window for (co, ho, wo)
    // Pack on-the-fly into registers/shared
    extern __shared__ uint64_t shmem[];
    uint64_t *window = &shmem[threadIdx.x * K_words];

    // ... pack K*K*C_in bits from input into window[K_words]

    // XNOR popcount accumulator
    int pc = 0;
    for (int k = 0; k < K_words; ++k)
        pc += __popcll(~(window[k] ^ weight_packed[co * K_words + k]));

    int raw = 2 * pc - K_bits;
    // padding correction omitted in sketch
    output[(co * H_out + ho) * W_out + wo] = alpha[co] * raw;
}
```

**Launch**:
```cuda
dim3 grid(C_out, H_out);
dim3 block(W_out);  // small (4, 8, 16, or 32 depending on layer)
size_t shmem_size = W_out * K_words * sizeof(uint64_t);
binary_conv2d_kernel<<<grid, block, shmem_size>>>(...);
```

### `batch_norm_2d_cuda`
```cuda
__global__ void bn2d_kernel(
    const float *x, const float *mean, const float *var,
    const float *gamma, const float *beta,
    float *y, int C, int HW, float eps
) {
    int c = blockIdx.x;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= HW) return;
    float inv = rsqrtf(var[c] + eps);
    float a = gamma[c] * inv;
    float b = beta[c] - mean[c] * a;
    y[c * HW + i] = a * x[c * HW + i] + b;
}
```

Trivial. RPReLU 유사.

### `rsign_cuda`, `rprelu_cuda`, `shortcut_cuda` — 동일하게 element-wise.

### FP32 stem — 옵션 2개
- **a. cuDNN**: `cudnnConvolutionForward` — 빠름, 라이브러리 활용
- **b. Custom**: 직접 작성, no dep

cuDNN 권장 (3×64×3×3 conv 1회만, 우리 모델에선 작은 영역).

---

## 5. Phase별 구현

| Phase | 작업 | 기간 |
|---|---|---|
| **1** | CUDA toolkit 셋업 (nvcc, cuda 12.x), 환경 체크 (`nvidia-smi`, `nvcc --version`) | 0.25d |
| **2** | 우리 CPU `xnor_kernel.c` → `xnor_kernel.cu`로 함수별 포팅 (트리비얼한 element-wise들) | 0.5d |
| **3** | `binary_conv2d` CUDA 커널 (L1, `__popc`) + 단위 테스트 | 1d |
| **4** | `resnet18_binary` host 코드 — kernel launch 시퀀스, 메모리 관리 | 0.5d |
| **5** | .bnn 파서 (CPU C 그대로) + cudaMalloc/cudaMemcpy로 GPU 전송 | 0.5d |
| **6** | CLI + 입력 로딩 + 결과 출력 | 0.5d |
| **7** | CPU C 결과와 bit-exact 검증 | 0.5d |
| **8 (옵션)** | L2: shared memory 최적화 | 1d |
| **9 (옵션)** | L3: BMMA tensor core 커널 | 2-3d |
| **10** | Profile (nsight compute), 발표 자료 | 0.5d |

### 텀프 현실: Phase 1-7 (3-4일) → L1 완성. L2/L3는 stretch.

---

## 6. Build system

### Makefile
```makefile
NVCC = nvcc
ARCH = -arch=sm_86          # RTX 3060 mobile
CFLAGS = -O3 -std=c++17 $(ARCH) -Iinclude
LDFLAGS = -lcudart -lcublas -lcudnn
SRCS = src/xnor_kernel.cu src/resnet18_binary.cu src/weights.cu src/main.cu
OBJS = $(SRCS:.cu=.o)

%.o: %.cu
	$(NVCC) $(CFLAGS) -c -o $@ $<

bnn_infer_cuda: $(OBJS)
	$(NVCC) $(CFLAGS) $(LDFLAGS) -o $@ $^
```

### 컴파일
```bash
make
./bnn_infer_cuda student.bnn cifar100_test.bin --max 1000 --device 0
```

---

## 7. 검증 전략

3 단계 cross-check:

1. **Single layer**: 같은 입력으로 CPU C와 CUDA C의 `binary_conv2d` 출력 비교
   - max_diff < 1e-4 (FP arithmetic 차이 허용 범위)
2. **End-to-end**: 같은 가중치/입력으로 PyTorch / CPU C / CUDA C 세 가지 logits 비교
   - argmax 일치, max_diff < 1e-3
3. **CIFAR-100 test set**: 전체 10000 sample 정확도 일치
   - 차이 < 0.1%p

---

## 8. 예상 성능 (RTX 3060 mobile, sm_86, 30 SMs, 6 GB)

| 레벨 | per-sample latency | throughput | 비고 |
|---|---|---|---|
| L1 (`__popc()`) | **~0.5-2 ms** | 500-2000 img/s | 메모리 대역폭 제한 |
| L2 (shared mem tiling) | ~0.3-0.8 ms | 1200-3000 img/s | +50% from L1 |
| L3 (BMMA tensor core) | **~50-200 μs** | **5000-20000 img/s** | INT1 가속 활용 |

비교:
- CPU C (1-thread): 22 ms — **L1 대비 ~20-40× 느림**
- CPU C (4-thread): 15 ms — **L1 대비 ~10-20× 느림**
- PyTorch CUDA FP32: 5-10 ms — **L1 비슷한 수준**
- **L3 BMMA**: PyTorch FP32 대비 **50-100× 빠름**

→ "RTX 3060에서 binary network가 PyTorch FP32보다 50-100× 빠름" 가능 (L3 완성 시)

---

## 9. 위험 요소

| Risk | 가능성 | 대응 |
|---|---|---|
| CUDA toolkit 환경 셋업 시간 | 중 | nvidia-driver, cuda-toolkit 사전 확인 |
| Padding correction CUDA 포팅 복잡 | 중 | CPU 버전 그대로 (host side) — kernel은 padding 없는 모드만 |
| BMMA 메모리 layout 까다로움 | 매우 높 (L3) | L3는 stretch goal, 텀프에선 L1까지 안전 |
| 작은 layer는 GPU launch overhead 큼 | 중 | L1 layer를 batch로 처리 (multiple samples 동시) |
| FP stem/FC가 작아 cuDNN 호출 overhead 큼 | 낮 | custom kernel로 대체 |
| 동기화 비용 (cudaDeviceSynchronize) | 낮 | 측정 시 careful, streamSync 사용 |

---

## 10. 발표 narrative 시나리오

### 시나리오 X — L1 완성 (3-4일)
> "BNN inference를 (1) CPU AVX C (2) RTX 3060 CUDA C로 풀 구현했습니다.
> Latency comparison (per-sample):
> - PyTorch FP32 CPU (BLAS): 11 ms
> - **Our C Binary (CPU, 4-thread)**: 15 ms
> - **Our CUDA C Binary (RTX 3060)**: 0.8 ms ← **18× faster than PyTorch CPU**
> - Theoretical INT1 Tensor Core (BMMA): 0.05 ms (RTX 3060), 0.01 ms (A100)
>
> 의존성: nvcc + CUDA runtime. 단일 binary."

### 시나리오 Y — L3 완성 (1주+)
> "RTX 3060의 **INT1 Tensor Core**를 활용한 BMMA 커널로 binary conv 가속:
> - Throughput: **20000 img/s** (PyTorch FP32 대비 50×)
> - Latency p50: **0.05 ms / sample**
> - GPU 활용률: 75-90%
> - 1.4 MB weights 전부 on-chip"

---

## 11. 권장 일정

학습 완료 가정 (D-day = 발표):

```
D-7 (오늘)      학습 안정화 (지금)
D-6 ~ D-5       발표 자료 초안 작성
D-5             CUDA toolkit 환경 셋업 + Phase 1-2
D-4             Phase 3 (binary_conv2d kernel L1)
D-3             Phase 4-5 (forward orchestration + weights)
D-2             Phase 6-7 (CLI + 검증) ← L1 데모 완성
D-1             발표 자료 마무리 + 리허설
D-day           발표
```

여유 있으면 L2/L3 추가, 없으면 L1으로 안정 마감.

---

## 12. CPU C → CUDA C 포팅 난이도

대부분 1-to-1 매핑:
- `for (int i = 0; ...)` → `int i = blockIdx.x * blockDim.x + threadIdx.x;`
- `malloc/free` → `cudaMalloc/cudaFree`
- `memcpy` → `cudaMemcpy`
- `__builtin_popcountll(x)` → `__popcll(x)` (CUDA intrinsic)
- 함수 → `__global__ void` 또는 `__device__`

**우리 CPU 커널은 이미 "포팅 직전 형태"**. ~70% 코드 재사용 가능.

---

## 13. 결론 & 권장

**Option A (Standalone CUDA C binary) + L1 (`__popc()`) 구현**

3-4일 작업으로:
- RTX 3060에서 실시간 binary inference 데모
- per-sample latency 0.5-2 ms
- CPU 데모 + GPU 데모 + Pareto 비교 → 발표 임팩트 최대화
- 의존성 nvcc만, 단일 binary

L2/L3 (shared mem + BMMA)는 stretch goal.

**현재 학습 진행 중 → 끝나고 시작. 학습 완료 후 4-5일 여유 있으면 충분히 가능.**

진행하시려면 Phase 1 (환경 셋업) 부터 도와드릴 수 있습니다.
