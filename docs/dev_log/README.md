# docs/dev_log/

개발 과정에서 생성된 계획 / 체크리스트 / 핸드오프 / 외부 모델 리뷰 모음.

최종 산출물은 아니지만, 의사결정 traceability를 위해 보관한다.

## Plans

- `plan_v1.md`, `plan_v2.md`, `plan_v4.md` — 학생 모델 / KD 학습 플랜의 버전별 evolution. v1 (초기) → v2 (Codex 리뷰 반영, RPReLU·double-skip·grad clip 추가) → v4 (1주일 일정, Track A/B 분리).
- `plan_avx_demo.md` — AVX CPU 추론 데모 설계 + Codex 리뷰.
- `plan_c_standalone.md` — C standalone 추론 (Python 의존 없는) 후속 검토.
- `plan_cuda_full.md` — CUDA 풀 구현 검토 (미구현, 향후 작업용).
- `plan_fpga_full.md` — FPGA 풀 구현 검토 (미구현).

## Checklists & handoff

- `gap_checklist_v1.md`, `gap_checklist_v2.md` — 발표·리포트 전 갭 분석.
- `HANDOFF_CUDA.md` — CUDA 작업자 인계용 문서.
- `pdf_diagnosis_v1.md` — 리포트 PDF 빌드 이슈 진단.
- `report_slides_architecture_v1.md`, `v2.md` — 리포트·슬라이드 구조 설계.

## External reviews

- `reviews/codex_review.md`, `codex_review.json` — Codex (GPT-5) 1차 코드 리뷰. LSQ 버그, EMA 비활성 권고, sign() 자명 트릭 등 지적.
- `reviews/gemini_review.json` — Gemini 리뷰 (부분).
