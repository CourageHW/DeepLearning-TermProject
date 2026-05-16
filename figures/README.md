# figures/

리포트·슬라이드용 그림 모음. 모두 `src/plotting/` 의 스크립트로 생성된다.

## Subdirectories

### `architecture/`
ReActNet block / overview / stem 아키텍처 다이어그램. drawio + 렌더링된 pdf/png/svg.

- `arch_reactnet.drawio`, `arch_reactnet_block.*`, `arch_reactnet_overview.*` — ReActNet block 상세 + 모델 overview.
- `arch_stem_step1.drawio`, `arch_stem_step1.*` — Stem (Conv1 path) 다이어그램.

생성: `python ../src/plotting/draw_arch.py`, `python ../src/plotting/draw_stem_step1.py`.

### `curves/`
학습곡선. `history/` 의 raw 데이터에서 생성.

- `curves_baseline.png` — student baseline 학습곡선.
- `curves_kd_effect.png` — KD on/off 비교.
- `curves_phase_b_ablation.png` — Phase B (skip / RPReLU) ablation 비교.
- `teacher_curves.png` — teacher fine-tune 곡선.

생성: `python ../src/plotting/plot_curves.py`.

### `bars/`
정량 비교용 막대 그래프 + activation 분포.

- `ablation_bars.png` — teacher 7-step ablation 정확도 향상.
- `latency_bars.png` — wall-time speedup 비교 (teacher fp32 vs student 1-bit, 1/4-thread).
- `activations.{pdf,png,svg}` — student RSign 입력 activation 분포 / sign 비율.

생성: `python ../src/plotting/plot_activations.py` (activations만), 나머지는 `report.md` 빌드 시 수동 생성.
