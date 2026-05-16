# scripts/

C 추론 demo 실행 셸 스크립트.

## Files

- **`run_student.sh`** — student (1-bit BNN) C 추론, 4-thread OpenMP, CIFAR-100 1000 이미지.
- **`run_teacher.sh`** — teacher (FP32 ResNet18) C 추론, baseline, 100 이미지.

## 사전 조건

1. `bash src/cpp_inference/build.sh` 로 `.so` 빌드 완료.
2. `git lfs pull` 로 `checkpoints/best_{student,teacher}.bnn` 받음.
3. CIFAR-100 데이터가 `~/data/cifar-100-python/` 에 있음.

## Run

```bash
# 저장소 루트에서
bash scripts/run_student.sh
bash scripts/run_teacher.sh
```
