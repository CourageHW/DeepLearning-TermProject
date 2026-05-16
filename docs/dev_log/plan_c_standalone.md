# Plan v5 — 순수 C 완전구현 (Standalone Binary)

Python orchestrator 없이 **C 단일 바이너리**로 학습된 W1A1 모델 추론.
HLS/FPGA로 가는 디딤돌이자, "진짜 C 추론 엔진" 발표 임팩트.

---

## 0. 목표

| 항목 | 현재 (Python+C 하이브리드) | 목표 (순수 C) |
|---|---|---|
| 인터페이스 | `python3 inference_c.py ...` | `./bnn_infer student.bnn testset.bin` |
| 의존성 | Python+PyTorch+torchvision+ctypes+numpy | **gcc + libc만** |
| Forward orchestration | Python loop (cell 11 형태) | **C `forward_resnet18_binary()` 함수** |
| 이미지 로딩 | torchvision.datasets.CIFAR100 | **flat binary file fread** |
| 출력 | terminal + numpy 처리 | **stdout text 또는 CSV** |
| 배포 | Python 환경 필요 | **단일 ELF binary** (FPGA HLS 가능) |

---

## 1. 디렉토리 구조 (신규)

```
.agents/avx_demo/c_only/
├── README.md              빌드 & 사용법
├── Makefile               빌드 (또는 build.sh)
│
├── include/
│   ├── xnor_kernel.h      primitive 선언 (기존 .c에서 분리)
│   ├── weights.h          .bnn loader API + struct ResNet18Weights
│   ├── resnet18_binary.h  forward 선언
│   └── cifar100_classes.h 클래스 이름 100개 (string array)
│
├── src/
│   ├── xnor_kernel.c      기존 primitives (xnor_gemm, binary_conv2d, BN 등)
│   ├── weights.c          .bnn 파일 파서, 메모리 매핑, 가중치 포인터 채우기
│   ├── resnet18_binary.c  forward_binary() — 전체 모델 orchestration
│   ├── resnet18_fp32.c    forward_fp32() — teacher 비교용
│   └── main.c             CLI, 입력 로딩, demo 루프, 출력
│
└── tools/
    ├── dump_testset.py    CIFAR-100 test → 단일 binary file (one-time)
    └── validate.py        Python ↔ C 출력 비교 (검증)
```

---

## 2. 데이터 구조 설계

### 2.1 가중치 컨테이너
```c
// include/weights.h
typedef struct {
    /* Stem (FP32) */
    float *conv1_weight;          // (64, 3, 3, 3)
    float *bn1_w, *bn1_b, *bn1_mean, *bn1_var;
    float *act_in_gamma, *act_in_beta, *act_in_prelu;

    /* 8 blocks × {bn1, rsign1, conv1(binary), bn2, act1, bn3, rsign2, conv2(binary), bn4, act2} */
    struct {
        float *bn1_w, *bn1_b, *bn1_mean, *bn1_var;
        float *rsign1_threshold;
        uint64_t *conv1_packed; float *conv1_alpha; int conv1_K_bits;
        int conv1_C_in, conv1_C_out, conv1_K, conv1_stride;
        float *bn2_w, *bn2_b, *bn2_mean, *bn2_var;
        float *act1_gamma, *act1_beta, *act1_prelu;
        /* branch 2 — same fields with suffix 2/3/4 */
        float *bn3_w, *bn3_b, *bn3_mean, *bn3_var;
        float *rsign2_threshold;
        uint64_t *conv2_packed; float *conv2_alpha; int conv2_K_bits;
        int conv2_C_in, conv2_C_out;
        float *bn4_w, *bn4_b, *bn4_mean, *bn4_var;
        float *act2_gamma, *act2_beta, *act2_prelu;
    } blocks[8];

    /* Tail */
    float *bn_out_w, *bn_out_b, *bn_out_mean, *bn_out_var;
    float *fc_weight, *fc_bias;

    /* Backing memory (single mmap or malloc'd block) */
    void *_blob;
    size_t _blob_size;
} ResNet18BinaryWeights;

int  load_bnn_student(const char *path, ResNet18BinaryWeights *out);
void free_bnn_student(ResNet18BinaryWeights *w);
```

같은 패턴으로 `ResNet18FP32Weights` (teacher).

### 2.2 Activation 버퍼 풀
```c
// 가장 큰 활성: layer1 output (64, 32, 32) = 65536 float = 256 KB
// 모든 단계 합쳐 < 1 MB. 정적 할당 가능.
typedef struct {
    float buf_a[64 * 32 * 32];    // 256 KB
    float buf_b[64 * 32 * 32];    // 256 KB (ping-pong)
    float buf_skip[64 * 32 * 32]; // skip connection 저장
    float pooled[512];
    float logits[100];
} ActivationScratch;
```
Forward에서 buf_a/buf_b 핑퐁으로 재사용 → 메모리 1 MB 미만.

### 2.3 .bnn 파일 포맷 (기존 그대로)
```
header: "BNN0" magic + uint32 num_tensors + uint32 reserved
tensors[]: name_length, name, dtype(0=fp32,1=u64,2=i32), ndim, shape[], raw_data
```

C 파서는 이름으로 dispatch:
```c
if (strcmp(name, "conv1.weight") == 0)         out->conv1_weight = (float*)data;
else if (sscanf(name, "layer%d.%d.bn1.weight", &s, &b) == 2) ...
```

---

## 3. Phase별 구현

### Phase 1 — 기존 코드 헤더 분리 (반나절)
- `xnor_kernel.c` → `include/xnor_kernel.h` + `src/xnor_kernel.c`
- 모든 함수에 prototype + 주석
- 컴파일 단위 분리 (`gcc -c`로 .o 따로)
- 기존 Python ctypes 호출은 그대로 동작 보장 (검증)

### Phase 2 — .bnn 파서 in C (반나절)
- `src/weights.c`: fopen/fread 기반 파서
- 이름→포인터 매핑 (struct field 채우기)
- mmap 옵션 (read-only): 가중치를 메모리 매핑 → 페이지 캐시 활용
- 에러 처리: magic 불일치, 누락된 텐서, shape mismatch

검증:
```c
ResNet18BinaryWeights w;
load_bnn_student("best_student.bnn", &w);
printf("conv1_weight[0..3] = %f %f %f %f\n", w.conv1_weight[0], ...);
// Python에서 출력한 값과 비교
```

### Phase 3 — Forward orchestration in C (1일)
- `src/resnet18_binary.c`:
  ```c
  void forward_binary(
      const ResNet18BinaryWeights *w,
      const float *input,          // (3, 32, 32) normalized
      ActivationScratch *scratch,
      float *logits                // (100,)
  );
  ```
- 흐름:
  1. fp32_conv2d(input, w->conv1_weight) → scratch->buf_a
  2. batch_norm_2d(buf_a, bn1) → buf_b
  3. rprelu(buf_b, act_in) → buf_a
  4. for b in 0..7: forward_block(w->blocks[b], buf_a, buf_b, scratch->buf_skip)
  5. batch_norm_2d(bn_out)
  6. adaptive_avgpool_1 → pooled
  7. linear_fp32(pooled, fc) → logits

- Block helper:
  ```c
  static void forward_block(const Block *b, float *x, float *y, float *skip, ...);
  ```

검증: 검증 스크립트로 Python ↔ C 출력 bit-exact 비교.

### Phase 4 — 입력 데이터 로딩 (반나절)
- 한 번만: `tools/dump_testset.py` 작성
  ```python
  # CIFAR-100 test → cifar100_test.bin
  # 10000 × (label_int32 + img_3×32×32_float32_normalized)
  # 총 ~120 MB
  ```
- C에서 fread로 순차 읽기:
  ```c
  FILE *f = fopen("cifar100_test.bin", "rb");
  fread(&label, 4, 1, f);
  fread(image, 4, 3*32*32, f);
  ```

대안: stb_image로 PNG 직접 디코딩 (의존성 ↑, 가치 낮음).

### Phase 5 — CLI + Demo Loop (반나절)
- `src/main.c`:
  ```c
  ./bnn_infer <weights.bnn> [test.bin] [--max N] [--threads T] [--ascii] [--compare other.bnn]
  ```
- 출력 포맷: Python 버전과 거의 동일
- `clock_gettime(CLOCK_MONOTONIC)`로 latency 측정
- ANSI 컬러 코드로 O/X 시각화

### Phase 6 — Build system (반나절)
- `Makefile`:
  ```makefile
  CC = gcc
  CFLAGS = -O3 -march=native -mpopcnt -mavx2 -fopenmp -Wall -Iinclude
  LDFLAGS = -lm -fopenmp
  OBJS = src/xnor_kernel.o src/weights.o src/resnet18_binary.o src/resnet18_fp32.o src/main.o
  bnn_infer: $(OBJS)
  	$(CC) $(LDFLAGS) -o $@ $^
  ```
- `make` → `./bnn_infer` 단일 바이너리

### Phase 7 — 검증 (반나절)
- `tools/validate.py`: PyTorch reference 출력 ↔ C binary 출력 logits 비교
- ASSERT: max_diff < 1e-4 (현재 Python+C도 bit-exact)
- 모든 10000 sample에 대해 argmax 일치 확인

### Phase 8 (선택) — FP32 Teacher in C
- `src/resnet18_fp32.c`: torchvision ResNet18 forward를 C로 (CIFAR stem 변형)
- 동일 컨테이너 패턴
- 같은 binary로 student / teacher 둘 다 추론 + 비교 가능
- "정확하게 fair 비교, 둘 다 C" 강조

---

## 4. 빌드 & 실행 (목표 모습)

```bash
# 1. Test set 한 번만 dump
python3 tools/dump_testset.py /path/to/CIFAR100 -o cifar100_test.bin

# 2. C 빌드
cd c_only && make

# 3. Student 단독 추론
./bnn_infer best_student.bnn cifar100_test.bin --color

# 4. Teacher 단독 추론
./bnn_infer best_teacher.bnn cifar100_test.bin --color

# 5. 둘 다 비교 (한 binary 안에서)
./bnn_infer best_student.bnn cifar100_test.bin --compare best_teacher.bnn --threads 4

# 6. 발표 demo: 100장만, ASCII 미리보기 포함
./bnn_infer best_student.bnn cifar100_test.bin --max 100 --ascii --compare best_teacher.bnn
```

---

## 5. 시간 예산

| Phase | 작업 | 시간 |
|---|---|---|
| 1 | 헤더 분리 + Makefile | 0.5d |
| 2 | .bnn 파서 in C | 0.5d |
| 3 | forward_binary 전체 | 1.0d |
| 4 | 입력 데이터 로딩 | 0.5d |
| 5 | CLI + demo loop | 0.5d |
| 6 | Build & link | 0.25d |
| 7 | Python vs C 검증 | 0.5d |
| **8 (옵션)** | FP32 teacher in C | 0.5d |
| **총** | | **~4 일** (옵션 포함 4.5일) |

---

## 6. 검증 전략

**3 단계 cross-check:**

1. **Per-primitive**: 각 primitive (fp32_conv2d, BN, RPReLU, binary_conv2d)를 isolated test로 PyTorch와 비교 (이미 부분적으로 됨)
2. **Layer-by-layer**: forward 중간 결과를 fp32 dump → Python에서 비교
3. **End-to-end**: 10000 sample 전체 logits → argmax 일치 + max_diff < 1e-4

---

## 7. 위험 요소 / Mitigation

| Risk | 가능성 | Mitigation |
|---|---|---|
| 가중치 이름 매칭 누락 | 중 | `if (!matched) fprintf(stderr, "unmatched: %s\n", name);` 디버그 출력 |
| BN buffer alignment (running_mean float32) | 낮 | memcpy로 안전 복사 (구조체 padding 회피) |
| 메모리 leak | 중 | valgrind 검증, free_bnn 호출 보장 |
| Padding correction 버그 (이미 해결) | 낮 | 현재 코드 그대로 활용, 검증 통과 |
| CIFAR-100 정규화 mismatch | 중 | dump_testset.py에서 같은 mean/std로 정규화 후 저장 |
| Python ↔ C float 정밀도 차이 | 낮 | fp32 통일, deterministic 연산만 사용 |
| Mmap 페이지 fault overhead | 낮 | 또는 fread로 전체 로드 (1.4 MB, 무시할 비용) |

---

## 8. 발표 임팩트 향상 포인트

| 항목 | 현재 (Python+C hybrid) | C standalone |
|---|---|---|
| "C 추론 엔진" 메시지 | △ (Python orchestration 보여야 함) | ✅ **단일 바이너리** |
| FPGA 변환 정당성 | "C 코드 일부 있음" | **"C 코드 그대로, HLS로 직접 변환 가능"** |
| 의존성 | Python + PyTorch + torchvision | **gcc만** |
| 배포 시 size | Python env GB 단위 | **<200 KB ELF + 1.4 MB weights** |
| Embedded 가능성 | ✗ | **✓** (예: RPi, MCU) |

발표 demo 시:
```bash
$ ldd bnn_infer
   linux-vdso.so.1
   libgomp.so.1   (OpenMP)
   libm.so.6
   libc.so.6
$ ls -lh bnn_infer
   -rwxr-xr-x  ~50 KB
```
"의존성 4개 (전부 표준 시스템 라이브러리), 50 KB 바이너리, **이게 binary inference 엔진**" — 강력.

---

## 9. 선택 항목

| 옵션 | 가치 | 시간 |
|---|---|---|
| AVX-512 VPOPCNTDQ branch (Ice Lake+) | ~+5× binary GEMM | 0.5d |
| ARM NEON 포팅 (모바일 demo) | 모바일 임팩트 | 1d |
| MMAP weights (메모리 효율) | 큰 모델 deployment | 0.25d |
| Streaming 입력 (stdin) | UNIX 파이프 친화 | 0.25d |
| TIFF/JPEG/PNG via stb_image | 임의 이미지 추론 | 0.5d |
| HLS pragma 추가 (FPGA 준비) | RTL 분석용 | 0.5d |

---

## 10. 다음 단계

1. **현재 학습 결과 (best_student.pth) 확보** ← 선행
2. 학습 완료 후 본 plan 실행:
   - Day 1: Phase 1-3 (헤더 분리 + .bnn 파서 + binary forward 골격)
   - Day 2: Phase 4-6 (입력 + CLI + 빌드)
   - Day 3: Phase 7 (검증) + Phase 8 (FP32 teacher)
   - Day 4: 발표 자료 보강

---

## 11. 비교 — Plan B (안 함)

만약 C 완전구현 안 하면:
- 발표는 Python+C 하이브리드로 충분
- 코드 양 ~600줄 줄어듦 (대신 Python orchestration 그대로)
- FPGA 변환 가능성 약간 약화 (Python 부분은 HLS 못 함)

**현재 Python+C hybrid 결과는 이미 충분히 강한 demo**. C standalone은 **+α 차별화** 목적. 1주일 일정에서 우선순위:
1. 학습 완료 + 정확도 확보 (절대 1순위)
2. 발표 자료 작성 (2순위)
3. **C standalone (3순위, 시간 남으면)**
4. RTL/FPGA HLS prototype (4순위)
