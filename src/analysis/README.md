# src/analysis/

학습된 모델의 사후 분석.

## Files

- **`eval_class_wise.py`** — CIFAR-100 100개 클래스별 top-1 accuracy 계산. teacher vs student per-class gap을 csv/png 로 출력.
- **`compute_bitops.py`** — ResNet18 vs A1W1ResNet18의 이론적 BitOps / FLOPs 비교. layer-wise breakdown 포함.

## Run

```bash
# 저장소 루트에서
python src/analysis/eval_class_wise.py --teacher checkpoints/best_teacher.pth \
                                       --student checkpoints/best_student.pth \
                                       --data ~/data --out figures/bars/classwise.csv

python src/analysis/compute_bitops.py
```
