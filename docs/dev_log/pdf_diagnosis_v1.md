# PDF 변환 문제 진단 v1

## 1. slides.pdf (Marp → PDF, 17 페이지) 문제

문제 패턴: **콘텐츠 overflow** — 마지막 줄이 footer ("KAU Deep Learning Term Project · 2026") 와 겹치거나 페이지 밖으로 잘림.

| Page | 슬라이드 | 증상 |
|---|---|---|
| 3 | Method Overview | "Loss: $L = ...$" 수식 + "(표준 Hinton KD, teacher가 reference)" 부연이 footer와 겹침. 마지막 단어 ("reference)") 가 footer 영역으로 밀려나감 |
| 4 | Student Architecture | 표 마지막 행 "α = E[\|W\|] (detached) / per-channel scale / ❌ (graph 차단)" 이 footer와 거의 겹침. 페이지 번호 4가 행 안에 들어가 있음 |
| 5 | Teacher Empirical Study | 마지막 sub-caption "Ablation = from-scratch 150 ep, Phase A = pretrained FT 80 ep — 학습 패러다임이 달라 ablation best 를 그대로 적용 불가." 가 footer와 **완전히 겹침**. 텍스트의 두 번째 줄 "By design: ReLU + mag 9 ..." 는 페이지 밖으로 잘림 |
| 7 | Quantitative Results | "📦 Memory: ..." 줄이 footer 영역으로 침범. "⚠️ ReActNet 격차 ..." 줄이 페이지 밖으로 잘림 |
| 9 | C Inference | 마지막 caption "Ratio 25 → 12× 감소 이유: ..." 두 번째 줄 "두 수치 모두 정직한 측정값." 이 footer와 겹침 |
| 14 | Backup B (Phase B) | 마지막 행 "Student init / teacher binary-호환 weight warm-start" 가 페이지 밖으로 잘려나감 |
| 15 | Backup C (Data Aug) | "Mixup / CutMix 수식" 의 수식 $\hat x = \lambda x_A + ...$ 이 footer와 거의 겹침 |
| 16 | Backup D (C Kernel) | 표 마지막 두 행 "shortcut_downsample / AvgPool + zero-pad" 와 "linear_fp32, adaptive_avgpool_1 / Classifier 등" 이 페이지 밖으로 잘림 |

OK 페이지: 1, 2, 6, 8, 10, 11, 12, 13, 17 (총 9장)
문제 페이지: 3, 4, 5, 7, 9, 14, 15, 16 (총 **8장**)

### 원인 추정 (slides)
- Marp 슬라이드 본문 영역 = 960×540 pts (16:9) 중 header/footer/margin 제외 본문 가용 높이는 약 440pt
- 본 슬라이드 일부에 5개 이상 항목 + 표 + 부연 + 수식 등이 동시에 들어가 footer 자리 침범
- 사용 폰트 크기 26px (기본) 또는 23px (일부 슬라이드 scoped style)
- footer 영역이 page-content와 분리되어 있지 않음 — 본문이 길어지면 footer 위로 흘러 들어감

## 2. report.pdf (pandoc + xelatex → PDF, 12 페이지) 문제

| # | 위치 | 증상 | 추정 원인 |
|---|---|---|---|
| **R1** | page 2 §2.2 Attention Transfer | `‖F‖₂` 의 `‖` 문자가 **빈 사각형 (□)** 으로 표시 | 한글 폰트 (Noto Sans CJK KR 추정) 에 `‖` (U+2016) glyph 미지원 |
| **R2** | page 5 §4.1 STE 설명 | `|x| ≤ 1` 의 `≤` 가 sub-script/공백으로 깨짐 ("backward 에서 \|x\| 1 외 영역") | 한글 폰트가 `≤` (U+2264) glyph 미지원 (수식 내부의 `\mathbf{1}_{\|x\| \le 1}` LaTeX 는 정상) |
| **R3** | page 6 §4.2 Block diagram | ASCII art (─, ┬, ├, └, ┐ box-drawing) 가 monospace 정렬을 잃고 단어 나열로 표시 | (a) 코드 펜스 미감지 또는 (b) box-drawing 문자가 monospace 폰트 미지원 |
| **R4** | page 8 §5.6 Data Aug 표 | "역할" 컬럼이 너무 좁아 Mixup 수식 cell ($\hat x = \lambda x_A + ...$) 이 4-5 줄로 wrap. RandAugment cell 도 매우 빽빽 | pandoc default 표는 컬럼 폭 자동 계산이 long-content 에 약함 |
| **R5** | page 10/11 §7.3 Latency 표 | 8-thread 행만 다음 페이지로 page break — 표가 두 페이지에 걸침 | pandoc default 페이지 break 가 표 중간을 자름 |

근본 원인:
- (a) `xelatex` 한글 폰트 (`Noto Sans CJK KR` 등) 가 일부 수학 기호 glyph 미지원
- (b) ASCII art 가 코드 펜스 안에 있더라도 box-drawing 문자가 monospace 폰트에서 깨짐
- (c) 표 컬럼 폭 자동 조정이 long-content 에서 균등 분배해 좁은 컬럼 발생
- (d) 표 안에 keepalive 설정 없어 page break 발생

## 3. 수정 방향 후보

### slides 수정 (4가지 옵션)

| 옵션 | 내용 | 효과 | 비용 |
|---|---|---|---|
| **S-A** | 문제 8장 콘텐츠 다이어트 (불필요 문구 / sub-caption 제거 또는 단축) | 가장 깔끔, 메시지 명확화 | 중 — 슬라이드별 검토 |
| **S-B** | 전역 CSS `section` 폰트 26px → 24px, line-height 1.4 → 1.3 으로 추가 압축 | 일괄 적용, 단순 | 저 — 가독성 약간 ↓ |
| **S-C** | 문제 슬라이드 분할 (예: slide 5 → 5a/5b) | 정보 손실 0 | 고 — 슬라이드 수 11→14 ↑, 발표 시간 영향 |
| **S-D** | Marp 전역 padding / footer 영역 재정의 (`style:` 에 `section { padding: 40px 60px; }` 등) | 가용 영역 확장 | 저 — 다른 페이지 layout 함께 변함 |

권장: **S-A + S-B 혼합** — 문제 페이지 콘텐츠 다이어트하고 폰트 약간 줄임. S-D 는 전 페이지에 영향 있어 신중.

### report 수정 (5가지 항목별)

| ID | 항목 | 수정안 |
|---|---|---|
| R1 | `‖F‖₂` | 수식 내부 `\|F\|` 로 변경 (단일 vertical bar 두 개) — `‖` U+2016 직접 사용 회피 |
| R2 | `|x| ≤ 1` (텍스트) | "backward 에서 `|x| <= 1` 외 영역" 으로 ascii `<=` 사용. 또는 수식으로 감싸 `$|x| \le 1$` |
| R3 | §4.2 Block diagram | ASCII art 를 ` ```text` 펜스로 명시. box-drawing 문자가 여전히 깨지면 `-`, `|`, `+` 만 사용한 단순 ASCII 로 대체 |
| R4 | §5.6 Mixup 표 cell | Mixup 의 수식을 표 cell 에서 빼고 표 아래 본문 단락으로 이동. 표는 "분류 / 기법 / 설정 / 역할 (1-2줄)" 만 유지 |
| R5 | §7.3 Latency 표 page break | (a) 표를 두 부분으로 분할 (8-thread 만 별도) 또는 (b) pandoc 변환 시 `--variable=longtable=true` 옵션 사용 — 표 자체는 markdown 그대로 유지 가능 |

## 4. 검토 요청 사항 (Codex / Gemini 동일)

1. 위 진단표 (slides 8장 + report 5항목) 중 빠진 문제가 있는가?
2. 각 옵션의 trade-off 평가. 특히 slides 의 S-A/S-B/S-C/S-D 중 어느 조합이 학부 발표용으로 가장 적절한가?
3. report R3 의 box-drawing 대체 — 단순 ASCII 가 정보 손실 없이 다이어그램을 표현 가능한가? 더 좋은 대안 (e.g. mermaid, image) 가 있는가?
4. 변환 명령에서 우리가 빠뜨린 옵션이 있는가? (font fallback, longtable, etc.)
