# CIFAR-100 1-bit Binary Neural Network with Knowledge Distillation

**Team 16 · KAU Deep Learning Term Project (2026-1)**

CIFAR-100 분류 task에 1-bit weight + 1-bit activation Binary Neural Network (BNN) 을 적용하고, XNOR + POPCNT 기반 C 추론 커널로 CPU 환경에서의 실측 speedup을 보인다.

---

## Results

| 항목 | Teacher (FP32 ResNet18) | Student (A1W1 ResNet18) |
|---|---|---|
| Top-1 accuracy | **82.44 %** | **69.06 %** |
| Model size | 44 MB | **1.9 MB** (≈ 23× 압축) |
| C inference wall-time (1-thread) | 1.00× | **21.21× faster** |
| C inference wall-time (4-thread, OpenMP) | 1.00× | **12.68× faster** |
| PyTorch ↔ C 출력 일치 | — | bit-exact (max diff = 0) |

- **Teacher**: ImageNet pretrained ResNet18을 CIFAR-100에 80 epoch fine-tune. 7-step ablation으로 baseline 75.74 % → 83.12 % → 최종 82.44 %.
- **Student**: ReActNet-lite (RSign / detached α / double-skip / RPReLU / AMP fp16-safe STE), Logit Knowledge Distillation (T=4, w=0.7).
- **C 추론 커널**: `src/cpp_inference/xnor_kernel.c` — binary conv, BN, RSign, RPReLU, FP32 stem/classifier, OpenMP 병렬. PyTorch와 출력 bit-exact.

상세 내용은 [`report/report.md`](report/report.md), 슬라이드는 [`slides/slides_native.pdf`](slides/slides_native.pdf) 참조.

---

## Directory layout

```
.
├── notebooks/        Kaggle 단일 세션 학습 노트북 (train.ipynb)
├── src/
│   ├── cpp_inference/   XNOR + POPCNT 기반 C 추론 커널 + Python 드라이버
│   ├── analysis/        class-wise eval, BitOps 계산
│   └── plotting/        학습곡선·아키텍처 다이어그램 생성 스크립트
├── scripts/          C 추론 demo 실행 셸 스크립트
├── docs/             프로젝트 헌장, 실험 로그, 발표 자료 원본
│   └── dev_log/      개발 과정 기록 (계획·체크리스트·핸드오프·외부 리뷰)
├── report/           최종 리포트 (Markdown + PDF)
├── slides/           최종 발표 슬라이드 (PowerPoint + PDF)
├── figures/          리포트·슬라이드용 그림 (architecture / curves / bars)
├── checkpoints/      학습된 weights (Git LFS)
└── history/          학습 곡선 raw 데이터 (CSV + pickle, LFS)
```

각 폴더에 자체 `README.md` 가 있다.

---

## Quick start

### 0. Clone (with LFS)

가중치·history pickle은 Git LFS로 추적된다.

```bash
git lfs install                      # 한 번만
git clone <repo-url>
cd Termp
git lfs pull                         # checkpoints/, history/*.pkl 받기
```

### 1. 학습 재현 (Kaggle T4)

`notebooks/train.ipynb` 를 Kaggle 노트북에 업로드. Cell 2의 Config 만 수정하면 됨. 단일 세션 (≈ 6h) 으로 teacher 80 ep + student 200 ep 완료.

### 2. C 추론 demo (Intel CPU, OpenMP)

```bash
bash src/cpp_inference/build.sh            # xnor_kernel.so, fp32_blas_kernel.so 빌드
bash scripts/run_student.sh                # student (1-bit, 4-thread) demo
bash scripts/run_teacher.sh                # teacher (FP32) baseline
```

CIFAR-100 데이터는 `~/data/cifar-100-python/` 에 있어야 한다 (`torchvision.datasets.CIFAR100(root='~/data', download=True)`).

---

## Git LFS 설정 (저장소 첫 push 시)

이 repo는 weights / pickle 을 LFS로 저장한다. 처음 push 하기 전:

```bash
git lfs install
git lfs track "*.pth" "*.bnn" "*.pkl"     # .gitattributes 에 이미 등록됨
git add .gitattributes
git add .
git commit -m "Initial commit: project structure + LFS"
git push -u origin main
```

`.gitignore` 는 `__pycache__/`, `*.so`, `.claude/`, `.codex/`, `.agents/`, `*.bak` 등을 제외한다.

---

## Tech stack

- **학습**: PyTorch 2.x, AMP fp16, Kaggle T4 GPU
- **Teacher**: torchvision ResNet18 (ImageNet pretrained)
- **Student**: ReActNet-lite custom impl. (`notebooks/train.ipynb` Cell 8-9)
- **C 추론**: C99 + GCC + OpenMP, XNOR + POPCNT, fp32 BLAS-naive 비교용
- **시각화**: matplotlib, python-pptx, drawio (architecture diagrams)

## Team

KAU 항공우주및기계공학부 · Deep Learning (2026-1) · Team 16

자세한 학술적 contribution은 [`report/report.md`](report/report.md)의 Abstract / §1.2 참조.
