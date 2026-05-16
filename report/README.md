# report/

최종 리포트.

## Files

- **`report.md`** — Markdown 원본. Abstract, Introduction, Related Work, Teacher empirical study (§3), Student design (§4), KD (§5), Hardware inference (§6), Results (§7), Discussion (§8), Conclusion.
- **`report.pdf`** — Pandoc / LaTeX 빌드 결과물.

## Re-build PDF

```bash
# pandoc + xelatex (한글 폰트 필요)
pandoc report.md -o report.pdf \
    --pdf-engine=xelatex \
    -V mainfont="Noto Serif CJK KR" \
    -V geometry:margin=25mm
```

PDF 빌드 이슈는 `../docs/dev_log/pdf_diagnosis_v1.md` 참조.
