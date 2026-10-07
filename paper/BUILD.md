# Build the prepared camera-ready paper

The PDF in this release is the prepared camera-ready revision, not a claim that
it has already replaced the reviewed OpenReview file. The reviewed PDF was V3.

From the repository root:

```bash
python -m pip install -r requirements-public.txt
python paper/reproduce_public.py
# Copy regenerated PNGs into paper/source if regenerating the paper itself.
cd paper/source
tectonic -X compile neurips_2026.tex --outdir ../build --keep-logs
```

Create the build directory first. Alternatively use a standard LaTeX/BibTeX
workflow. The source has the official final workshop options, author block,
funding statement, and no modifications to the official style.

Official style SHA-256:
`C3FC2894E83D2517CA18B66741D6C595986D97957DC08EC08BB2125A7EC4555A`.
Downloaded from https://tai-eval.github.io/files/neurips_2026.sty on 2026-10-07.

Verified PDF: ten content pages, references on pages 11-12, appendix from page
12, checklist from page 15, 21 pages total. All fonts embedded; no unresolved
references or out-of-page text. PDF SHA-256:
`76AA4F296E10F0C7D73D7DB90F1B1D4D55E31D1DD39E44EB3E4D8FED21A8843B`.
