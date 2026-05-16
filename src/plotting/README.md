# src/plotting/

리포트·슬라이드용 figure 생성.

## Files

- **`plot_curves.py`** — `../../history/` 의 학습곡선 pickle / csv 를 읽어 train·val loss / acc 곡선을 그린다 (`figures/curves/*.png`).
- **`plot_activations.py`** — student 의 RSign 입력 activation 분포 / sign 비율 히스토그램 생성 (`figures/bars/activations.{png,pdf,svg}`).
- **`draw_arch.py`** — ReActNet block / overview 다이어그램 생성. 출력: `figures/architecture/arch_reactnet_*`.
- **`draw_stem_step1.py`** — Stem (Conv1 → BN → ReLU → MaxPool) step-1 다이어그램.
- **`build_editable_pptx.py`** — 마크다운 (`docs/slides.md`) 을 편집 가능한 pptx로 빌드. 발표용 native pptx 생성에 사용.

## Run

```bash
# 저장소 루트에서
python src/plotting/plot_curves.py        # → figures/curves/
python src/plotting/plot_activations.py   # → figures/bars/activations.*
python src/plotting/draw_arch.py          # → figures/architecture/
python src/plotting/draw_stem_step1.py    # → figures/architecture/
python src/plotting/build_editable_pptx.py  # → slides/slides_editable.pptx
```

## 의존성

`matplotlib`, `numpy`, `python-pptx`, `Pillow`. drawio 파일은 [drawio desktop](https://www.drawio.com/) 으로 열어 추가 편집 가능.
