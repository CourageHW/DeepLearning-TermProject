# notebooks/

Kaggle 단일 세션 학습 노트북.

## Files

- **`train.ipynb`** — Teacher (ResNet18) fine-tune + Student (A1W1 ReActNet-lite) KD 학습. Cell 0–12 의 13개 셀로 구성, Cell 2의 Config만 수정하면 전체 파이프라인 재실행 가능.

## Cell map

| Cell | 역할 |
|---|---|
| 0 | Imports |
| 1 | Kaggle 경로, device 설정 |
| 2 | **Config — 모든 하이퍼파라미터** |
| 3 | `set_seed` |
| 4 | `CIFAR100Dataset` (RAM 캐시) |
| 5 | DataLoader / GPU transforms |
| 6 | `mixup`, `cutmix`, `train_epoch`, `evaluate`, Feature KD 헬퍼 |
| 7 | `build_teacher`, `make_optimizer` |
| 8 | Binary primitives (`BinaryActivationSTE`, `RSignActivation`, `ShortcutDownsample`) |
| 9 | `RPReLU`, `A1W1BinaryConv2dV2`, `A1W1BasicBlockV2`, `A1W1ResNet18v2`, `init_student_from_teacher` |
| 10 | Teacher 학습 루프 (80 ep) |
| 11 | Student 학습 루프 (200 ep, KD) |
| 12 | 결과 저장 (`best_*.pth`, history pickle) |

## 실행 환경

- Kaggle Notebook · T4 GPU (16 GB)
- PyTorch ≥ 2.0, torchvision, AMP fp16
- 데이터: Kaggle competition `26-deep-learning-course` 또는 `torchvision.datasets.CIFAR100`

학습이 끝나면 weights 는 `../checkpoints/` 로, history 는 `../history/` 로 이동하면 됨.
