# Current camera-ready aggregate outputs

These files contain only aggregate numerical evidence, not clinical records,
subject/episode identifiers, diagnosis codes, timestamps, or individual risks.

- `tae_submission_analysis.py` generates metric, calibration, weighting,
  multiplicity, seed-overlap, and strictly pre-prediction repetition summaries.
- `characterize_selected_episodes.py` generates discordant-selection profiles,
  final ensemble set counts, and exact stress membership exchanges.
- `matched_stability_and_shift_profiles.py` generates fixed-ensemble-size
  overlap references and largest-5%-shift group profiles.
- `audit_summary.json` records the full retained-package verification.
- `current_contracts_verification.json` records validation by the hardened
  current ensemble code without changing the fitted probabilities.

See ../PAPER_OUTPUT_MAP.md for the final table/figure mapping. Candidate age is
time since first mention, not diagnosis span. The largest-shift groups are
pipeline-specific, post-hoc, and conditional on stress eligibility.

Offline public checks and plots use `python paper/reproduce_public.py` from the
repository root. Regenerating row-level summaries or bootstrap intervals
requires the authorized retained private inputs.
