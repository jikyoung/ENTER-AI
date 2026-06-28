# Analysis Scripts

This directory contains product and data-quality validation scripts used while
repositioning ENTER-AI toward the Review Signal Agent direction.

The generated outputs under `analysis/out/` are intentionally ignored because
they may contain local experiment data or large CSV files. The important
findings are summarized in `docs/project_history.md` and
`docs/sparkclaw_submission.md`.

## Key Scripts

- `day0_audit.py`: audits URL uniqueness, keyword hit rate, short text ratio,
  source distribution, and noisy categories.
- `check_url_uniqueness.py`: checks whether crawled records can support
  citation-level evidence.
- `check_googleplay_metadata.py`: validates Google Play metadata availability.
- `review_action_demo.py`: tests a no-LLM review-to-action MVP flow with recent
  Google Play reviews.
