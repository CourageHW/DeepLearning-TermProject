# Handoff — CUDA C 구현 (학교 노트북 RTX 3060)

> 이 문서는 **새 Claude Code 세션**이 학교 노트북(RTX 3060)에서 CUDA 구현을 이어갈 수 있도록 작성된 자기완결형 핸드오프.

---

## 0. 컨텍스트 한 줄

학습된 W1A1 ResNet18 (Top-1 ~68.72%, CIFAR-100)을 **RTX 3060 mobile**에서 추론하는 **CUDA C standalone binary**를 만들어야 함.

---

## 1. 현재 프로젝트 상태 (학교 노트북 시작 시점)

### 완료된 것 ✅
- A1W1ResNet18v2 모델 정의, 학습 파이프라인 (`train.ipynb`)
- Teacher 학습: `best_teacher.pth` (ResNet18 FP32, val 82.44%)
- Student 학습: `best_student.pth` (W1A1, val 68.72%, **재학습 결과 확인 필요**)
- **CPU C 추론 파이프라인** 전체 (`/.agents/avx_demo/`):
  - `xnor_kernel.c` — 모든 primitives (binary GEMM, BN, RPReLU, FP conv 등)
  - `export_weights.py` — `.pth` → `.bnn` 변환
  - `inference_c.py` — Python orchestrator, validate/demo/compare/compare3 모드
  - `models.py` — PyTorch reference (검증용)
  - OpenMP multi-thread 지원
  - PyTorch와 bit-exact 검증 통과
- 폴리시 완료:
  - CIFAR-100 클래스 이름, 메모리 배너, latency bar, ASCII 이미지, Top-3, class-wise breakdown, confusion table, three-way compare

### 진행 중 ⏳
- Student 재학습 (Feature KD off 후 안정화 결과 확인)

### 다음 작업 (이 문서의 목표) 🎯
- **CUDA C 풀 구현** (학교 노트북 RTX 3060에서)
- 결과: `bnn_infer_cuda` 단일 binary, RTX 3060에서 추론

---

## 2. 필수 사전 준비 (학교 노트북에서)

### 2.1 환경 체크
```bash
# GPU 확인
nvidia-smi
# 예상 출력: RTX 3060 (mobile), driver 525+

# CUDA toolkit
nvcc --version
# 예상: 12.x

# 없으면 설치:
# https://developer.nvidia.com/cuda-downloads (Linux)
# WSL2면 https://docs.nvidia.com/cuda/wsl-user-guide/
```

### 2.2 프로젝트 동기화
git 사용 시:
```bash
git pull origin <branch>
```
또는 USB/rsync로 `/home/yonggi/KAU/3-1/딥러닝/Termp/` 전체 복사.

### 2.3 학습된 가중치 확보
다음 두 파일 필수:
- `best_student.pth` (학습 완료 후 Kaggle output에서 다운로드)
- `best_teacher.pth` (이미 있음)

프로젝트 루트에 둘 것.

### 2.4 Python 환경 (PyTorch 검증용)
```bash
pip install torch torchvision numpy
```

---

## 3. 작업 디렉토리 구조

```
Termp/
├── .agents/
│   ├── HANDOFF_CUDA.md           ← 이 문서
│   ├── plan_cuda_full.md         ← 상세 plan (참고)
│   ├── PROJECT.md (../)          ← 전체 프로젝트 정리
│   ├── avx_demo/                 ← CPU C 구현 (참고용)
│   │   ├── xnor_kernel.c         ★ CUDA로 포팅할 대상
│   │   ├── inference_c.py        ★ Python orchestrator (검증 참고)
│   │   ├── models.py             ★ PyTorch reference
│   │   ├── export_weights.py     ★ .bnn 포맷 (그대로 재사용)
│   │   └── ...
│   └── cuda_demo/                ← 신규 작업 디렉토리 (만들어야 함)
│       ├── src/
│       │   ├── xnor_kernel.cu
│       │   ├── resnet18_binary.cu
│       │   ├── weights.cu
│       │   └── main.cu
│       ├── include/
│       │   └── *.h
│       └── Makefile
├── best_student.pth
├── best_teacher.pth
└── train.ipynb
```

---

## 4. Claude Code 새 세션 시작 시 — 첫 메시지 추천

학교 노트북에서 새 Claude Code 세션 열면 다음 메시지로 시작:

> "CIFAR-100 BNN 텀프 CUDA C 구현 이어서 진행. 컨텍스트는 `.agents/HANDOFF_CUDA.md`와 `.agents/plan_cuda_full.md`에 있음. 현재 RTX 3060 학교 노트북. Phase 1 환경 셋업부터 시작."

Claude는 이 두 문서를 읽으면 전체 컨텍스트 확보 가능.

---

## 5. Phase별 작업 체크리스트

`plan_cuda_full.md` 섹션 5 참고. 요약:

### Phase 1 — 환경 셋업 (0.25일)
- [ ] `nvidia-smi` GPU 확인
- [ ] `nvcc --version` CUDA toolkit 확인
- [ ] `mkdir -p .agents/cuda_demo/{src,include}` 디렉토리 생성
- [ ] Hello world `.cu` 컴파일 테스트 (`nvcc test.cu -o test`)

### Phase 2 — Primitives 포팅 (0.5일)
- [ ] `xnor_kernel.cu` 생성, CPU 버전 헤더 복사
- [ ] Element-wise 커널 포팅:
  - [ ] `batch_norm_2d` → `__global__ void bn2d_kernel`
  - [ ] `rsign` → element-wise kernel
  - [ ] `rprelu` → element-wise kernel
  - [ ] `add_inplace`, `relu_inplace` → element-wise
- [ ] 각 커널 launch wrapper 함수 (host)

### Phase 3 — `binary_conv2d` CUDA 커널 (1일) — **핵심**
- [ ] L1: scalar `__popcll()` 기반 커널
- [ ] 1 thread = 1 output position
- [ ] Shared memory에 im2col window 캐싱
- [ ] Padding correction 처리
- [ ] α scaling
- [ ] 단위 테스트: random input + CPU 결과 비교

### Phase 4 — `forward_resnet18_binary` host 코드 (0.5일)
- [ ] `resnet18_binary.cu` — kernel launch 시퀀스
- [ ] 활성 버퍼 (cudaMalloc), ping-pong
- [ ] 8 binary blocks 오케스트레이션
- [ ] stem (FP32 conv, cuDNN 또는 custom) + tail (BN + avgpool + FC)

### Phase 5 — `.bnn` 파서 + GPU 전송 (0.5일)
- [ ] `weights.cu` — CPU C 파서 재활용
- [ ] `cudaMalloc` + `cudaMemcpyHostToDevice`로 가중치 GPU로 전송
- [ ] `ResNet18BinaryWeightsGPU` struct

### Phase 6 — CLI + Demo (0.5일)
- [ ] `main.cu` — argparse 비슷한 처리
- [ ] CIFAR-100 test set 로딩 (이미 `dump_testset.py`가 만든 binary 사용)
- [ ] `cudaEvent_t` 로 latency 측정
- [ ] `[O/X]` 출력

### Phase 7 — 검증 (0.5일)
- [ ] CPU C 출력과 logits bit-similar 비교 (max_diff < 1e-3)
- [ ] argmax 100% 일치
- [ ] 10000 test sample 정확도 일치 (±0.1%p)

### Phase 8 (옵션) — L2/L3 최적화
- [ ] Shared memory tiling
- [ ] WMMA BMMA Tensor Core (`bmma_sync`)

### Phase 9 — 발표 자료
- [ ] CPU vs CUDA latency 비교 차트
- [ ] Throughput 측정 (`nsys profile` 또는 단순 wall-clock)

---

## 6. 주요 코드 매핑 — CPU C → CUDA C

이미 우리 CPU 커널은 GPU 포팅 친화적. 매핑:

| CPU 패턴 | CUDA 패턴 |
|---|---|
| `for (int i = 0; i < N; i++)` | `int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < N)` |
| `malloc(N)` / `free` | `cudaMalloc` / `cudaFree` |
| `memcpy(dst, src, N)` | `cudaMemcpy(dst, src, N, cudaMemcpyHostToDevice)` |
| `__builtin_popcountll(x)` | `__popcll(x)` (CUDA intrinsic) |
| OpenMP `#pragma omp parallel for` | kernel launch with block/grid |
| 함수 `void f(...)` | `__global__ void f_kernel(...)` |

---

## 7. 검증 워크플로우 (필수)

각 Phase 끝날 때마다:

```bash
# 1. CPU 데모로 ground truth 만들기
cd .agents/avx_demo
python3 inference_c.py --demo --max 100 --weights best_student.bnn > cpu_output.txt

# 2. CUDA 데모로 비교
cd ../cuda_demo
./bnn_infer_cuda ../best_student.bnn ../cifar100_test.bin --max 100 > cuda_output.txt

# 3. diff
diff cpu_output.txt cuda_output.txt
# 예상: argmax 100% 일치
```

또는 Python 스크립트로 logits 직접 비교:
```python
import numpy as np
cpu_logits = np.load('cpu_logits.npy')
cuda_logits = np.load('cuda_logits.npy')
print(f"max_diff: {np.abs(cpu_logits - cuda_logits).max()}")
print(f"argmax match: {(cpu_logits.argmax(-1) == cuda_logits.argmax(-1)).mean()}")
```

---

## 8. 잠재적 문제 + 해결

| 문제 | 원인 | 해결 |
|---|---|---|
| `nvcc: command not found` | CUDA toolkit 미설치 | https://developer.nvidia.com/cuda-downloads |
| `CUDA error: invalid device function` | sm_86 미컴파일 | Makefile에 `-arch=sm_86` 명시 |
| Padding correction 결과 다름 | CPU 로직과 차이 | CPU `compute_pad_correction_one`을 host에서 한 번 계산 후 GPU로 전송 |
| 작은 layer 매우 느림 | kernel launch overhead | Batch 처리 (여러 sample 동시) |
| BMMA 메모리 layout 디버그 | 128-bit align 실패 | `cudaMallocPitch` 또는 명시적 padding |
| OOM | RTX 3060 mobile 6GB | Batch size 줄임 |

---

## 9. 빠른 참고 — `xnor_kernel.cu` 시작 템플릿

```cuda
// .agents/cuda_demo/src/xnor_kernel.cu
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void binary_conv2d_kernel(
    const float    *input_signed,
    const uint64_t *weight_packed,
    const float    *alpha,
    float          *output,
    int C_in, int H, int W, int C_out,
    int K, int stride, int padding,
    int K_bits, int K_words,
    int H_out, int W_out
) {
    int co = blockIdx.x;
    int wo = blockIdx.y * blockDim.x + threadIdx.x;
    int ho = blockIdx.z;
    if (co >= C_out || wo >= W_out || ho >= H_out) return;

    // 1. Pack im2col window for (ho, wo)
    extern __shared__ uint64_t shmem[];
    uint64_t *window = shmem + threadIdx.x * K_words;
    for (int k = 0; k < K_words; k++) window[k] = 0;

    for (int ci = 0; ci < C_in; ci++) {
        for (int kh = 0; kh < K; kh++) {
            int ih = ho * stride - padding + kh;
            for (int kw = 0; kw < K; kw++) {
                int iw = wo * stride - padding + kw;
                int bit_idx = (ci * K + kh) * K + kw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    float v = input_signed[(ci * H + ih) * W + iw];
                    if (v >= 0.0f)
                        window[bit_idx / 64] |= (uint64_t)1 << (bit_idx % 64);
                }
            }
        }
    }

    // 2. XNOR + popcount
    int pc = 0;
    const uint64_t *w = weight_packed + co * K_words;
    for (int k = 0; k < K_words; k++)
        pc += __popcll(~(window[k] ^ w[k]));

    int raw = 2 * pc - K_bits;
    // TODO: padding correction (host-side precomputed)
    output[(co * H_out + ho) * W_out + wo] = alpha[co] * raw;
}

// Host wrapper
extern "C" void binary_conv2d_launch(
    const float *d_input, const uint64_t *d_weight, const float *d_alpha,
    float *d_output,
    int C_in, int H, int W, int C_out, int K, int stride, int padding,
    int K_bits, int K_words
) {
    int H_out = (H + 2*padding - K) / stride + 1;
    int W_out = (W + 2*padding - K) / stride + 1;
    int block = 32;
    dim3 grid(C_out, (W_out + block - 1) / block, H_out);
    size_t shmem = block * K_words * sizeof(uint64_t);
    binary_conv2d_kernel<<<grid, block, shmem>>>(
        d_input, d_weight, d_alpha, d_output,
        C_in, H, W, C_out, K, stride, padding, K_bits, K_words, H_out, W_out
    );
    cudaDeviceSynchronize();
}
```

---

## 10. Makefile 시작 템플릿

```makefile
# .agents/cuda_demo/Makefile
NVCC = nvcc
ARCH = -arch=sm_86
CFLAGS = -O3 -std=c++17 $(ARCH) -Iinclude
LDFLAGS = -lcudart

SRC_DIR = src
OBJ_DIR = obj
SRCS = $(wildcard $(SRC_DIR)/*.cu)
OBJS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRCS))

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(CFLAGS) -c -o $@ $<

bnn_infer_cuda: $(OBJS)
	$(NVCC) $(LDFLAGS) -o $@ $^

$(OBJ_DIR):
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) bnn_infer_cuda
```

---

## 11. 발표용 예상 결과 (RTX 3060 L1 기준)

| 백엔드 | per-sample latency | throughput |
|---|---|---|
| PyTorch FP32 CPU (BLAS, 4-thread) | ~11 ms | ~90 img/s |
| **Our C Binary (CPU, 4-thread)** | ~15 ms | ~67 img/s |
| **Our CUDA C Binary (RTX 3060, L1)** | ~0.5-2 ms | **500-2000 img/s** |
| (옵션 L3 BMMA, 1주+) | ~0.05 ms | ~20000 img/s |

→ **CPU binary 대비 10-30× speedup. PyTorch FP32 대비 5-20× speedup.**

---

## 12. 발표 narrative (CUDA L1 완성 시)

> "동일한 binary ResNet18 모델을 3가지 백엔드로 비교:
>
> 1. **PyTorch FP32 (CPU, multi-thread BLAS)**: 11 ms/sample — 산업 표준 베이스라인
> 2. **Our C Binary (CPU AVX, 4-thread)**: 15 ms/sample — 동일 C runtime에서 PyTorch와 비슷, 알고리즘 차이만
> 3. **Our CUDA C Binary (RTX 3060 mobile, custom kernel)**: 0.8 ms/sample — **PyTorch CPU 대비 14× 빠름**
>
> CUDA 커널은 100% 자체 작성, BMMA Tensor Core 활용 시 추가 10×+ 가능.
> 의존성: nvcc + CUDA runtime. 단일 binary."

---

## 13. 시간 예산 (학교 노트북 시작 가정)

| Day | 작업 | 결과 |
|---|---|---|
| **1** | Phase 1-2: 환경 + element-wise primitives | 컴파일 동작 확인 |
| **2** | Phase 3: `binary_conv2d` 커널 + 단위 테스트 | 단일 layer 검증 통과 |
| **3** | Phase 4-5: forward + weights | 전체 forward 동작 |
| **4** | Phase 6-7: CLI + 검증 | **L1 데모 완성** |
| 5 (옵션) | Phase 8: shared mem 최적화 | L2 |
| 6-7 (옵션) | BMMA tensor core | L3 |

---

## 14. 기존 자산 — 그대로 재사용

### 14.1 .bnn 포맷 (재사용)
`.agents/avx_demo/export_weights.py`가 만든 `best_student.bnn` 그대로 사용. 포맷 변경 X.

### 14.2 검증용 CPU 데모 (재사용)
`.agents/avx_demo/inference_c.py --demo` 출력이 ground truth.

### 14.3 PyTorch reference (재사용)
`.agents/avx_demo/models.py`로 logits 비교 가능.

### 14.4 Test set binary 만들기 (선택)
PyTorch 의존성 없이 데모하려면:
```python
# .agents/cuda_demo/tools/dump_testset.py (만들면 됨)
import numpy as np
from torchvision import datasets
from torchvision.transforms import v2 as T
import torch

MEAN = np.array([0.4850, 0.4560, 0.4060]).reshape(3, 1, 1)
STD  = np.array([0.2290, 0.2240, 0.2250]).reshape(3, 1, 1)
ds = datasets.CIFAR100(root='~/data', train=False, download=True,
                       transform=T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]))
with open('cifar100_test.bin', 'wb') as f:
    for img, label in ds:
        img_norm = ((img.numpy() - MEAN) / STD).astype(np.float32)
        f.write(np.int32(label).tobytes())
        f.write(img_norm.tobytes())
print(f"wrote {len(ds)} samples")
```

---

## 15. Claude Code 사용 팁 (학교 노트북에서)

1. **세션 시작 시 첫 메시지**: 섹션 4 참고
2. **Plan 자료 참조**:
   - `.agents/HANDOFF_CUDA.md` ← 이 문서 (워크플로우)
   - `.agents/plan_cuda_full.md` ← 상세 기술 plan
   - `PROJECT.md` ← 프로젝트 전체 컨텍스트
3. **CPU 코드 참고**: `.agents/avx_demo/xnor_kernel.c`를 참조해 CUDA로 변환
4. **검증**: 매 Phase마다 CPU 데모 출력과 비교
5. **diff 발생 시**: 우선 padding correction 확인 → 다음 alpha → 다음 BN running stats

---

## 16. 후속 작업 (선택)

L1 완성 후 추가하면 좋은 것:
- [ ] L2 shared memory tiling
- [ ] L3 BMMA tensor core (`bmma_sync`)
- [ ] `nsys profile`로 성능 분석
- [ ] PyTorch C++ extension으로 wrap (`torch.utils.cpp_extension`)
- [ ] FPGA HLS prototype (Vitis HLS) — Plan v6 참고

---

## 17. 핵심 정보 빠른 참조

| 정보 | 값 |
|---|---|
| 모델 | A1W1ResNet18v2 (RPReLU + double-skip, 99% binary) |
| 가중치 | `best_student.pth` (~11.07M params, 학습된 W1A1) |
| Teacher | `best_teacher.pth` (FP32 ResNet18, 82.44%) |
| .bnn 파일 | export_weights.py로 변환 |
| CPU 데모 (reference) | `.agents/avx_demo/inference_c.py --demo` |
| 정규화 | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
| 입력 shape | (3, 32, 32) float32 |
| 출력 | logits (100,) float32 → argmax = class |
| 타깃 GPU | RTX 3060 mobile, sm_86, 6 GB VRAM |
| 타깃 latency | < 1 ms per sample (L1) |
| 타깃 throughput | > 1000 img/s (L1) |

---

## 18. 결론

학교 노트북 첫 세션 흐름:

```
1. nvidia-smi / nvcc --version 환경 체크
2. 프로젝트 동기화 (git pull 또는 USB)
3. Claude Code 세션 시작:
   "HANDOFF_CUDA.md와 plan_cuda_full.md 참고. Phase 1부터 시작."
4. Day 1: 환경 + element-wise primitives
5. Day 2-4: binary_conv2d 커널 + 전체 forward
6. Day 4 끝: L1 데모 완성, 발표 자료 보강
```

**자기 컴퓨터에서 만든 자산 그대로 활용 + GPU만 학교 노트북. 효율적 작업 분담.**

발표 임팩트 확보 + 학습된 모델을 GPU에서 실시간 추론 시연 가능.
