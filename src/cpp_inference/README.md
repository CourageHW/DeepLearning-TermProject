# src/cpp_inference/

XNOR + POPCNT 기반 1-bit BNN C 추론 커널 + Python 드라이버. PyTorch 결과와 **bit-exact** 일치 (max diff = 0).

## Files

| 파일 | 역할 |
|---|---|
| `xnor_kernel.c` | 핵심 binary conv (XNOR + POPCNT), BN, RSign, RPReLU, FP32 stem/classifier, OpenMP 병렬. |
| `fp32_blas_kernel.c` | FP32 ResNet18 (teacher) C 추론 — speedup 비교 baseline. |
| `build.sh` | 위 두 파일을 shared library (`*.so`) 로 빌드. GCC + `-march=native -fopenmp -O3`. |
| `models.py` | Python측 ResNet18 / A1W1ResNet18 정의 (export_weights용 reference). |
| `export_weights.py` | PyTorch `.pth` → C 추론용 binary `.bnn` 포맷 변환. |
| `inference_c.py` | ctypes로 `.so` 호출, CIFAR-100 evaluation + benchmark 드라이버. |
| `demo_inference.py` | 단일 이미지에 대한 forward 시각화 / 디버깅용. |
| `validate_and_bench.py` | PyTorch ↔ C 출력 bit-exact 검증 + wall-time 측정. |
| `bench_3way.py` | (teacher fp32 / student fp32 / student 1-bit) 3-way 벤치. |
| `bench_blas_vs_naive.py` | FP32 baseline 내에서 BLAS vs naive 비교. |

## Build

```bash
bash build.sh
# → xnor_kernel.so, fp32_blas_kernel.so 생성 (gitignored)
```

요구: `gcc` ≥ 9, OpenMP, AVX2 지원 CPU (Intel Core 8세대+ / AMD Zen+ 이상 권장).

## Weights

가중치는 `../../checkpoints/` 에 있다 (LFS). C 추론용 `.bnn` 포맷이 필요하면:

```bash
python export_weights.py --pth ../../checkpoints/best_student.pth \
                         --out ../../checkpoints/best_student.bnn
```

(이미 변환된 `best_student.bnn`, `best_teacher.bnn` 이 checkpoints/ 에 포함되어 있음.)

## Run

```bash
# 저장소 루트에서
bash scripts/run_student.sh        # 1-bit, 4-thread, 1000 images
bash scripts/run_teacher.sh        # FP32 baseline, 100 images

# 또는 직접
python inference_c.py --demo --thread 4 \
    --weights ../../checkpoints/best_student.bnn \
    --data ~/data --max 1000
```

## Verification

```bash
python validate_and_bench.py --pth ../../checkpoints/best_student.pth \
                              --bnn ../../checkpoints/best_student.bnn
# 출력: max abs diff = 0.0 (bit-exact), wall-time 비교 출력
```
