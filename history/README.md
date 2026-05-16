# history/

학습 곡선 raw 데이터. `src/plotting/plot_curves.py` 의 입력.

## Layout

### `teacher/`
- `history_tch_baseline.csv` / `.pkl` — teacher fine-tune 80ep 의 epoch별 train/val loss·acc.

### `student/`
- `history_stu_baseline.csv` / `.pkl` — student 최종 baseline (KD 포함, full skip + RPReLU).
- `history_stu_e1_kd0.{csv,pkl}` — Ablation: KD 가중치 0 (KD off).
- `history_stu_e2_singleskip.{csv,pkl}` — Ablation: double-skip → single-skip.
- `history_stu_e3_no_rprelu.{csv,pkl}` — Ablation: RPReLU → PReLU.

각 `.csv` 는 epoch / train_loss / train_acc / val_loss / val_acc / lr 컬럼.  `.pkl` 은 같은 데이터의 pickle (plot 스크립트가 dict 형태로 사용).

## LFS

`.pkl` 파일은 `.gitattributes` 에 따라 Git LFS로 추적된다. `.csv` 는 일반 텍스트.

## Plot

```bash
python src/plotting/plot_curves.py    # → figures/curves/
```
