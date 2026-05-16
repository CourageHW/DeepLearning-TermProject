# Plan v6 — FPGA 완전 구현 (RTL + Synthesis + Deployment)

A1W1ResNet18v2 W1A1 binary network의 **풀 FPGA 구현**.
RTL 설계 → 합성 → 시뮬레이션 → 실제 보드 deployment.

> **솔직 평가**: 풀 구현은 **2-4개월** 프로젝트 (FPGA 경험자 기준).
> 텀프 1주일이면 **HLS 프로토타입 + 합성 리포트**까지가 현실. 그래도 전체 플랜은 아래.

---

## 0. 목표 정의

| 단계 | 결과물 | 시간 | 텀프 가능? |
|---|---|---|---|
| **L1. 분석만** | 자원 추정표, 슬라이드 | 2일 | ✅ |
| **L2. HLS 1-layer prototype** | binary_conv2d HLS C++ + 합성 리포트 | 3-5일 | △ 빡빡 |
| **L3. HLS 전체 네트워크** | 모든 layer HLS, simulation 통과 | 2주 | ❌ |
| **L4. RTL + Verilog 풀 구현** | SystemVerilog 모듈 + testbench | 1-2개월 | ❌ |
| **L5. 보드 deployment** | PYNQ/ZCU102에서 실시간 추론 | 추가 2-4주 | ❌ |

**현실적 텀프 deliverable: L1 + L2**.
L3-L5는 후속 연구/졸업 프로젝트 범위.

---

## 1. 타깃 FPGA 선택

| 보드 | 칩 | LUTs | BRAM | DSPs | 가격 | 우리 모델 fit? |
|---|---|---|---|---|---|---|
| **PYNQ-Z2** | Zynq XC7Z020 | 53K | 4.9 Mb | 220 | $200 | ❌ weights 너무 큼 (11.2 Mb 필요) |
| **Ultra96-V2** (ZU3EG) | Zynq US+ | 70K | 7.6 Mb | 360 | $250 | △ weights 절반만 on-chip, DDR도 사용 |
| **KV260** (K26) | Zynq US+ | 117K | 26 Mb | 1248 | $500 | ✅ 안정적 |
| **ZCU102** (ZU9EG) | Zynq US+ | 274K | 32.1 Mb | 2520 | $$$ (대학 eval) | ✅ 여유 |
| **Alveo U50** | Virtex US+ | 870K | 28 Mb | 5952 | DC급 | ✅ 압도적 |

**권장 타깃**: **KV260** 또는 **ZCU102** (학교 보유 가능성).
시뮬레이션만 한다면 보드 없이도 Vivado/Vitis 무료 라이선스로 가능.

---

## 2. 아키텍처 설계

### 2.1 Top-level 데이터플로우

```
                 ┌─────────────────────────────────────────────┐
                 │                  KV260 / ZCU102              │
DDR     ────►   │  ┌──── PS (ARM A53) ──────┐                   │
(weights)        │  │  Linux + driver        │                   │
                 │  │  DMA controller        │                   │
                 │  └──────┬───────────────────┘                 │
                 │         │ AXI4                                 │
                 │  ┌──────▼───────────────────────────────────┐ │
                 │  │            PL (FPGA fabric)              │ │
                 │  │  Input ──► Stem ──► L1 ──► L2 ──► L3 ──► L4 ──► AvgPool ──► FC ──► Output  │ │
                 │  │              (FP)   (binary blocks ×8)              (FP)         │ │
                 │  │                                                                   │ │
                 │  │  Weights BRAM (on-chip)  +  Activation buffers (ping-pong)        │ │
                 │  └──────────────────────────────────────────┘ │
                 └─────────────────────────────────────────────┘
```

### 2.2 Layer-wise 자원 추정 (KV260 기준)

| Stage | 연산 | 주요 자원 | DSP | LUT 추정 | BRAM (Mb) |
|---|---|---|---|---|---|
| Input quant | uint8 → fp/int8 | 약간의 LUT | 0 | ~500 | 0 |
| **Stem conv** (3×3, 3→64) | FP32 conv | DSP 다수 | 64 (time-mux) | ~3K | 0.1 |
| BN1 + RPReLU | per-channel scale | DSP for fp | 32 | ~2K | 0.05 |
| **Binary block ×8** | XNOR + popcount + α | LUT 위주 | 8 (α only) | ~8K each = **64K** | 6.0 (weights) |
| BN/RPReLU × 33 | 잡다 fp ops | DSP | 64 | ~6K | 0.1 |
| Shortcut × 3 | AvgPool + ch pad | LUT | 0 | ~1K | 0 |
| AvgPool out | 4×4 mean | LUT | 0 | ~500 | 0 |
| **FC** (512 → 100) | FP32 linear | DSP | 100 (or time-mux) | ~5K | 0.2 |
| Control/AXI | DMA, FSM | LUT/FF | 0 | ~5K | 0 |
| **총** | | | **~250 DSP** | **~95K LUT** | **~6.5 Mb** |

KV260 capacity (117K LUT / 26 Mb BRAM / 1248 DSP) → **~80% LUT, ~25% BRAM, ~20% DSP**. ✅ Fits.

### 2.3 Pipeline / Parallelism 전략

**Layer pipelining** (모든 layer가 동시에 다른 sample 처리):
- Pipeline depth = 17 stages
- Throughput = 1 sample per N cycles (N = 최대 stage latency)
- 예상 throughput: **~5000-20000 img/s** @ 200 MHz

**Loop unrolling** (per-layer):
- Stem: K×K = 9 multiplies/cycle (1 output pixel = 9 cycles)
- Binary conv: 64 popcounts/cycle (분기 8개 동시), full unroll 가능
- BN: per-channel parallel (64-512 wide)

**Tradeoff**: Throughput ↑ ↔ 자원 사용 ↑. Time-multiplexing으로 자원 줄임.

### 2.4 메모리 계층

| 데이터 | 위치 | 크기 |
|---|---|---|
| Input image (3×32×32) | URAM/BRAM | 12 KB |
| Stem weights | BRAM | 7 KB |
| Binary conv weights (packed) | BRAM | **1.4 MB** ← 핵심 |
| BN/RPReLU params | BRAM | ~200 KB |
| FC weights | BRAM or DDR | 200 KB |
| Activation buffers (ping-pong) | BRAM | ~500 KB |
| **총 BRAM** | | **~2.5 MB = 20 Mb** ← KV260 fits |

---

## 3. 구현 경로 비교

### 3.1 FINN automated flow (가장 표준)

```
PyTorch (Brevitas) → ONNX → FINN compiler → HLS C++ → Vitis HLS → Bitfile
```

장점:
- 자동화. 모델 정의만 하면 나머지 자동
- Xilinx Research 검증된 toolchain
- Throughput estimator 내장

단점:
- Brevitas로 **재학습 필요** (양자화 정의를 Brevitas API로 표현)
- FINN가 지원하는 layer만 가능 (RPReLU는 표준 아님 → 직접 wrapping 필요)
- Tool 학습 곡선

### 3.2 HLS 직접 작성 (권장)

기존 `xnor_kernel.c`를 거의 그대로 활용:
```cpp
// vitis_hls/binary_conv2d_hls.cpp
#include "hls_stream.h"
#include "ap_int.h"

void binary_conv2d_hls(
    hls::stream<ap_int<8>>  &input_stream,   // ±1 quantized input
    ap_uint<64>             *weight_packed,  // BRAM
    float                   *alpha,
    hls::stream<float>      &output_stream
) {
    #pragma HLS INTERFACE m_axi port=weight_packed bundle=W
    #pragma HLS INTERFACE axis port=input_stream
    #pragma HLS INTERFACE axis port=output_stream
    #pragma HLS PIPELINE II=1

    // 이하 우리 xnor_gemm 로직 거의 동일
    // ap_uint<64> xor_val = ~(av ^ wv);
    // int pc = __builtin_popcount(xor_val);
    // ...
}
```

장점:
- **C 코드 이미 있음** → 변환 비용 낮음
- 빠른 합성 + 시뮬레이션
- 자원 리포트 즉시 확인

단점:
- 직접 토폴로지 설계 (dataflow, BRAM partition)
- Stream interface 학습 필요

### 3.3 Manual RTL (Verilog)

장점: 최적 성능, 풀 컨트롤
단점: 풀 코드 수천 줄, 디버깅 매우 어려움

**텀프 범위 외**.

---

## 4. Phase별 구현 — L2 (HLS prototype) 기준

### Phase 1 — Vitis HLS 환경 셋업 (0.5일)
- Xilinx Vitis HLS 2024.x 무료 설치 (Linux 권장)
- 프로젝트 생성, 타깃 보드 (KV260) 설정
- "Hello world" HLS C++ 합성 확인

### Phase 2 — `binary_conv2d` HLS 변환 (1일)
- 기존 `binary_conv2d` 함수를 HLS C++로 포팅
- `ap_uint<64>` 타입으로 weight/activation 표현
- `hls::stream` 또는 array interface
- HLS pragmas:
  ```cpp
  #pragma HLS PIPELINE II=1
  #pragma HLS UNROLL factor=8
  #pragma HLS ARRAY_PARTITION variable=weight_packed dim=1 type=complete
  ```
- Synthesis 후 자원 리포트:
  - LUT 사용량
  - DSP 사용량
  - Latency (cycles)
  - Throughput (II = Initiation Interval)

### Phase 3 — Co-simulation으로 정확도 검증 (0.5일)
- HLS C/RTL co-simulation
- 입력: 우리 `inference_c.py`의 binary_conv2d 출력
- 비교: HLS 출력 vs C 출력 bit-exact 여부
- ASSERT: max_diff == 0

### Phase 4 — 다른 primitive HLS 변환 (1-2일)
- `fp32_conv2d` (stem) — DSP fully utilized
- `batch_norm_2d` — fixed-point or fp16
- `rprelu`
- `linear_fp32` (FC)

각 primitive 합성 후 자원 누적 표 작성.

### Phase 5 — Synthesis 리포트 분석 (0.5일)
- 전체 estimated LUT/DSP/BRAM 합산
- KV260 capacity 대비 utilization %
- Critical path → 예상 max frequency
- Estimated end-to-end latency @ 200 MHz

### Phase 6 — 발표 자료 작성 (0.5일)
- 자원 추정 표
- 합성 리포트 스크린샷
- "C 커널 → HLS C++ → RTL" 변환 demonstration
- Throughput / Power 추정치
- 향후 작업 (FPGA deployment) 로드맵

**L2 합계: ~3-4일**

---

## 5. Phase별 구현 — L3-L5 (참고용)

### L3. HLS 전체 네트워크 (+1주)
- Top-level dataflow function (`forward_resnet18_binary`)
- Layer 간 stream interface
- Memory partition (BRAM 분배)
- Full network co-simulation
- 합성 → 자원 합계
- **bitstream 생성 (Vivado)**

### L4. Manual RTL (참고만)
- SystemVerilog로 각 layer 모듈 작성
- Testbench (Vivado XSIM 또는 Verilator)
- 통합 + timing closure

### L5. 보드 deployment (+2-4주)
- Petalinux SDK 또는 PYNQ 환경 셋업
- DMA driver + Python frontend
- 실시간 demo: 카메라 → FPGA → 분류 결과

---

## 6. 도구 / 라이선스

| 도구 | 용도 | 라이선스 |
|---|---|---|
| **Vitis HLS** | HLS C++ → RTL | 무료 (학생) |
| **Vivado** | RTL → bitstream | 무료 (학생) |
| **Vitis** | SW + HW 통합 | 무료 (학생) |
| **Verilator** | RTL 시뮬레이션 | 오픈소스 |
| **FINN** | PyTorch → FPGA 자동 | 오픈소스 (Apache) |
| **Brevitas** | PyTorch quantization for FINN | 오픈소스 |

PyTorch → Brevitas 재학습 안 하면 FINN 사용 어려움. 우리는 HLS 직접 갈 수 있음.

---

## 7. 검증 전략

### 단계별 cross-check

1. **HLS C/C++ simulation** ↔ 우리 Python `inference_c.py` 출력
   - 같은 가중치, 같은 입력, bit-exact 일치
2. **HLS C/RTL co-simulation** ↔ HLS C 결과
   - Vitis 자체 검증, 보통 bit-exact
3. **Vivado functional simulation** ↔ co-sim 결과
   - timing 포함
4. **실제 보드 hardware-in-the-loop** ↔ Vivado 결과
   - L5 단계, 텀프 외

각 단계마다 ASSERT(max_diff == 0).

---

## 8. 위험 요소 / 현실 점검

| Risk | 가능성 | 영향 | 대응 |
|---|---|---|---|
| FPGA 경험 부족 → tool 학습 곡선 | 매우 높 | 매우 큼 | L2 prototype에 집중, L3+는 후속 |
| HLS pragma 미스로 자원 폭증 | 중 | 큼 | 단계별 합성, 단일 layer부터 |
| BRAM 부족 → DDR fallback | 낮 | 중 | KV260 정도면 fits |
| Critical path → low frequency | 중 | 중 | Pipeline depth 늘리기 |
| FP32 stem/FC가 자원 잡아먹음 | 중 | 중 | int8 quantization fallback 옵션 |
| RPReLU floating-point overhead | 중 | 중 | fp16 또는 fixed-point 변환 |
| **시간 부족** | **매우 높** | 매우 큼 | L1 분석 + L2 prototype만 발표 |

---

## 9. 발표 narrative 강화 (L2까지 했을 때)

> "우리 binary inference 엔진은 **C → HLS C++ → RTL** 변환 경로를 검증했습니다.
> Vitis HLS로 단일 binary conv layer를 합성한 결과, **KV260 (Zynq UltraScale+)** 보드에서:
> - LUT: 약 8K (전체의 7%)
> - DSP: 0 (binary는 곱셈기 불필요)
> - BRAM: 80 KB
> - II=1, latency 100 cycles (∼500 ns @ 200 MHz)
>
> 전체 ResNet18 binary inference는 KV260 단일 칩에 fit하며,
> 추정 throughput **5000 img/s @ 2-5W**.
> 같은 모델을 CPU에서 추론하면 ~45 img/s @ 65W → **100× throughput, 13× 전력 효율**."

(숫자는 실제 합성 결과로 대체)

---

## 10. 시간 예산 매트릭스

| 결과물 | 최소 시간 | 텀프 가능? | 발표 임팩트 |
|---|---|---|---|
| L1 분석 슬라이드만 | 2일 | ✅ | 중 |
| **L1 + L2 prototype** | **4일** | **△ (학습 마무리 후)** | **상** |
| L3 풀 네트워크 HLS | 2주 | ❌ | 매우 상 |
| L4 RTL 풀 구현 | 1-2개월 | ❌ | 최상 |
| L5 보드 deployment | 2-4개월 | ❌ | 최상 |

---

## 11. 권장 결정

학습 끝나고 발표 1주일 남았다 가정:

**Option A (안전)** — L1만:
- 자원 추정표 + 아키텍처 다이어그램 + 향후 작업
- 발표 슬라이드 2-3장 추가
- **2일 작업**
- "FPGA-ready" 메시지 충분히 전달

**Option B (도전, 추천)** — L1 + L2:
- 위 + Vitis HLS prototype (binary_conv2d 1개 layer 합성)
- 실제 자원 숫자 확보
- **4일 작업**
- "C 코드 → 실측 RTL 합성 검증" 강력한 narrative

**Option C (욕심)** — L3 시도:
- L1+L2 + 풀 네트워크 HLS 시도
- 부분 완성도 OK
- 2주+ 필요 — **텀프 일정 초과 가능**

---

## 12. 권장 작업 순서 (Option B 기준)

```
학습 완료 →
   Day 1: Vitis HLS 환경 셋업 + 우리 binary_conv2d 함수 포팅
   Day 2: HLS C/RTL co-simulation 검증
   Day 3: 다른 primitives (BN, RPReLU, FP conv) HLS 변환
   Day 4: 합성 + 자원 리포트 + 발표 슬라이드
```

---

## 13. 후속 작업 로드맵 (참고)

텀프 끝나고 시간 있을 때:
1. L3: 풀 네트워크 HLS + bitstream (1-2주)
2. PYNQ 환경 셋업 + DMA driver (1주)
3. L5: 실제 보드에 deploy + webcam demo (2-3주)
4. 논문/발표 → 학회 (CVPR/ICCV 워크샵, FPGA 학회 등)

---

## 14. 결론

**1주일 텀프에서 진짜 "완전 구현"은 비현실적**.
하지만 **L1 (분석) + L2 (HLS prototype)** 까지면:
- "FPGA에서 동작 가능함을 합성 리포트로 입증"
- 우리 C 코드가 "FPGA-ready"임을 demonstrate
- 학술적/실무적 정당성 확보

진짜 풀 구현은 **졸업 프로젝트 / 인턴 연구 / 학회 논문**의 범위.

본 plan은 그 로드맵을 명시적으로 보여주는 문서이기도 함.
