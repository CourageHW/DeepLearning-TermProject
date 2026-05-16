# Plan v3 — AVX CPU Binary Inference Demo

## 0. 한 줄 목적
T4 GPU엔 INT1 native HW가 없어 W1A1 student의 진짜 speedup을 못 보여줌. **x86 CPU의 POPCNT + AVX2/AVX-512**로 XNOR-popcount 커널을 만들어 FP32 대비 실측 speedup 그래프를 발표에 박는다.

## 1. Why CPU, not GPU
- T4 (Turing): INT4/INT8 Tensor Core ✓, **INT1 X** → fp16 conv로 fallback, speedup 거의 없음
- A100/H100 (Ampere/Hopper): INT1 BMMA ✓ but Kaggle은 T4만 무료
- 모든 x86 CPU (Kaggle Xeon 포함): **POPCNT 명령 + AVX2** 표준. AVX-512 VPOPCNTDQ는 Ice Lake+ 한정
- 모바일/엣지 BNN 배포의 **실제 타깃이 CPU/MCU**라는 점도 발표 narrative에 적합

## 2. Scope (Level별 stop 가능)

### Level 1 — Single Layer Benchmark (필수, ~4h)
- 가장 큰 binary conv 1개 선택: `layer4.1.conv2` (512×512×3×3) — GEMM으로 reshape 후 (M=64, N=512, K=4608)
- 측정:
  - FP32 GEMM: `torch` CPU 또는 `np.matmul` (BLAS 가속)
  - Binary GEMM: 우리 C 커널 (XNOR + `__builtin_popcountll`)
- 출력: latency 단일 숫자, speedup ratio (예: 7×)
- 검증: 같은 weight/input으로 PyTorch `F.conv2d(sign(x), sign(w))` 결과와 bit-exact 일치

### Level 2 — Full ResNet18 Forward (선택, +4h)
- ResNet18 binary path 전체 C로 구현 (BN/RPReLU는 fp32 그대로, conv만 우리 커널)
- FP32 teacher 전체 forward vs Binary student 전체 forward 비교
- 출력: end-to-end latency 표, throughput (images/sec)

### Level 3 — AVX-512 차별 비교 (선택, +2h)
- CPU feature detection (`/proc/cpuinfo` 또는 `cpuid`)
- AVX-512 VPOPCNTDQ 사용 버전 컴파일 (`-mavx512vpopcntdq -mavx512bw`)
- AVX2 vs AVX-512 throughput 비교 그래프

## 3. Implementation Components

### 3.1 C 커널 (`xnor_kernel.c`, ~150줄)
```c
// 핵심 API
void xnor_gemm(
    const uint64_t* A,   // [M × K_words]  packed binary input (im2col 결과)
    const uint64_t* B,   // [N × K_words]  packed binary weight (out_ch first)
    int32_t* C,          // [M × N]        output sign-dot product
    int M, int N, int K_words
);

// im2col + binarize + pack helper
void im2col_pack(
    const float* in,     // [B × C × H × W]
    uint64_t* out,       // packed bits
    int B, int C, int H, int W,
    int K, int stride, int padding,
    int H_out, int W_out
);
```

핵심 inner loop:
```c
uint64_t xnor = ~(A[ik] ^ B[jk]);
pc += __builtin_popcountll(xnor);
C[ij] = 2 * pc - K_bits;   // sign-dot product
```

컴파일:
```
gcc -O3 -march=native -mavx2 -mpopcnt -fopenmp -shared -fPIC xnor_kernel.c -o xnor.so
```

### 3.2 Python wrapper (notebook cell, ~80줄)
- `ctypes.CDLL('./xnor.so')`
- numpy → uint64 packing utility
- Layer-level forward 호출
- timeit + statistics (median, p95)

### 3.3 Weight extraction
- `best_student.pth` 로드 → `state_dict()`
- 각 binary conv layer의 `.weight` 추출 → `sign()` → bit packing
- 결과: 각 layer마다 `uint64` 배열

### 3.4 Validation cell
1. Toy 입력 (10×3×8×8) 만들기
2. PyTorch forward: `student(toy_input)` 의 layer4.1.conv2 출력 hook
3. 우리 C 커널 forward: 같은 입력을 binarize + pack → xnor_gemm → unpack
4. Element-wise 비교: 정확히 일치해야 함 (binarize 후 dot product는 결정적)

## 4. Inputs / Outputs

**Inputs:**
- `best_student.pth` (학습 완료된 A1W1ResNet18v2)
- `best_teacher.pth` (FP32 비교용)
- 테스트 입력: `(1, 3, 32, 32)` (단일 이미지 latency) + `(64, 3, 32, 32)` (배치 throughput)

**Outputs:**
- Latency table (markdown)
- Bar chart: FP32 vs Binary, layer별 / 전체
- 표 컬럼: Layer, M, K, FP32 ms, Binary ms, Speedup
- (Level 2) Total forward: FP32 X ms / Binary Y ms / speedup Z×
- (Level 3) AVX2 vs AVX-512 추가 컬럼

## 5. 예상 수치 (검증 전 추정)

| Layer | M | N | K | FP32 (ms) | Binary AVX2 (ms) | 예상 speedup |
|---|---|---|---|---|---|---|
| layer4.1.conv2 (512→512, 3×3) | 64 | 512 | 4608 | ~4-6 | ~0.5-1.0 | 6-10× |
| layer3.1.conv2 (256→256, 3×3) | 256 | 256 | 2304 | ~3-5 | ~0.4-0.8 | 5-8× |
| End-to-end ResNet18 forward | — | — | — | ~30-50 | ~5-12 | 4-8× |

AVX-512 시 추가 ~1.5-2× 더.

## 6. Risks / Mitigations

| Risk | 가능성 | 대응 |
|---|---|---|
| Kaggle CPU가 AVX-512 미지원 | 중 | AVX2 baseline 우선 (모든 Kaggle Xeon ≥ AVX2) |
| `im2col` 구현 버그 | 중 | `torch.nn.functional.unfold`로 reference 비교 |
| ctypes 메모리 정렬/타입 mismatch | 낮 | `np.ascontiguousarray(.astype(np.uint64))` 강제 |
| Sign convention 혼동 ({-1,+1} vs {0,1}) | 중 | validation cell에서 elementwise 비교로 catch |
| GCC autoVec이 POPCNT loop vectorize 못 함 | 낮 | `__builtin_popcountll` 명시 → POPCNT instr 강제 |
| OpenMP race condition | 낮 | row-parallel만 (M축 분할), N/K는 single-thread |
| Kaggle CPU 코어 수 적음 (4 core) → 멀티스레드 효과 작음 | 중 | 옵션. 우선 single-thread 발표 |

## 7. Open Questions for Codex Cross-Check

1. **Kernel formula 정확성**: `dot_product = 2*popcount(XNOR) - K_bits`가 sign({-1,+1}) dot product와 일치하는지? 신호/순서 혼동 없는지?
2. **Realistic speedup**: AVX2 POPCNT 처리량(per cycle) vs FP32 GEMM (BLAS, vectorized) — 예상 비율 sanity check
3. **im2col 필요성**: 작은 spatial dim (8×8, 4×4) layer에서도 im2col 오버헤드 < direct conv 이득인가?
4. **Memory layout**: A=(M, K_words), B=(N, K_words) — N행을 inner loop 하는 게 cache-friendly한지 (B 전치 필요?)
5. **K not multiple of 64**: padding 처리 — input channel 수가 64 배수 아닌 layer (첫 conv 64 in)는 어떻게?
6. **PyTorch FP32 CPU 비교의 공정성**: numpy `@` 는 MKL/OpenBLAS 가속 받음. 우리 단일 thread C 커널이 multi-thread BLAS와 비교되면 불공평. 둘 다 single-thread로 고정 권장?
7. **검증 numerics**: ResNet18에서 BN 통계와 RPReLU γ/β가 fp32라 우리 binary conv 출력은 PyTorch와 같아야 함. 정확히 일치 안 하면 어디 의심?
8. **기존 라이브러리 대안**: larq compute engine, bitorch-engine 같은 게 더 빠를 텐데 우리 발표 목적에 PyTorch 통합 비용이 너무 큰가?
9. **AVX-512 실효성**: VPOPCNTDQ vs AVX2 POPCNT — Ice Lake 이상에서만 의미 있는데 Kaggle 환경 검증 방법?
10. **잠재적 함정**: ctypes 호출 오버헤드, 작은 layer는 ctypes call cost > 실제 compute가 될 수 있음. 어디부터 의미 있는 측정?

## 8. Timeline

| 단계 | 시간 |
|---|---|
| C 커널 작성 (Level 1) | 2-3h |
| Python wrapper + weight packing | 1h |
| Validation (PyTorch vs C 결과 일치) | 1-2h |
| Layer-level benchmark + 그래프 | 1h |
| **Level 1 총** | **5-7h** |
| Level 2 (full forward) | +4h |
| Level 3 (AVX-512 비교) | +2h |
| **풀 스택** | **11-13h** (≈2일) |

## 9. Deliverables

1. `xnor_kernel.c` — C source
2. `demo_avx_inference.ipynb` — wrapper + benchmark + 그래프
3. `benchmark_results.csv` — 표 데이터
4. `speedup_chart.png` — bar chart for presentation
5. 발표 슬라이드 1-2장 (실측 speedup 그래프 + narrative)
