# checkpoints/

학습된 모델 가중치. **Git LFS** 로 추적된다.

## Files

| 파일 | 크기 | 설명 |
|---|---|---|
| `best_teacher.pth` | 44 MB | FP32 ResNet18 fine-tune (top-1 82.44 %) |
| `best_student.pth` | 44 MB | A1W1 ResNet18 student (top-1 69.06 %), KD 학습 |
| `best_teacher.bnn` | 44 MB | teacher의 C 추론용 binary 포맷 |
| `best_student.bnn` | 1.9 MB | student의 1-bit packed 포맷 (≈ 23× 압축) |

`.pth` 는 PyTorch state_dict, `.bnn` 은 `src/cpp_inference/export_weights.py` 로 변환된 C-loader 포맷이다.

## LFS

```bash
git lfs install            # 한 번만
git lfs pull               # checkpoints 받기
```

LFS 미설치 상태로 clone 하면 위 파일들이 작은 pointer 텍스트로 들어온다.

## 재생성

학습:
```
notebooks/train.ipynb  →  best_*.pth
```

C 추론용 binary 변환:
```bash
python src/cpp_inference/export_weights.py \
    --pth checkpoints/best_student.pth \
    --out checkpoints/best_student.bnn
```
