# Camera-ready output map

The dated camera-ready source is the canonical manuscript. Labels, rather than
table numbers, are used here because moving reviewer-responsive tables changes
numbering.

| Paper label or analysis | Aggregate input / implementation |
|---|---|
| `tab:primary` and all ensemble metrics | `aggregate_outputs/recomputed_ensemble_metrics.csv` |
| Primary and component paired contrasts | `aggregate_outputs/multiplicity_sensitivity.csv` |
| Figure `fig:decision_gap` | `analysis/generate_decision_gap_figure.py` reads primary contrasts and set overlap |
| Figure `fig:evaluation_profile` | `analysis/generate_evaluation_profile_figure.py` reads `evaluation_conclusion_matrix.csv` and strict stress bootstrap |
| Calibration diagnostics and reliability curves | `calibration_summary.csv`, `calibration_bins.csv`; `tae_submission_analysis.py` |
| Ensemble selected-set overlap | `ensemble_top10_cross_representation.csv` |
| Matched-size retraining reference | `matched_size_stability.csv`; `matched_stability_and_shift_profiles.py` |
| Discordant selection profile | `discordant_top_decile_characterization.csv`; `characterize_selected_episodes.py` |
| Strict repetition shifts/intervals | `copy_forward_strict_pre_prediction_*.csv`; `tae_submission_analysis.py` |
| Stress membership changes | `full_repetition_top_decile_churn.csv`; `characterize_selected_episodes.py` |
| Largest-shift episode profile | `stress_shift_tail_characterization.csv`; `matched_stability_and_shift_profiles.py` |

Five calibrated probabilities are averaged for each primary ensemble. Selection
uses `ceil(0.10 * N)` and ascending row identifier to break equal-risk ties.
Matched-size analysis averages one or two calibrated seed outputs; within-
representation two-seed groups are disjoint. Same seed numbers across
representations are index matches, not identical random trajectories. Pair
distributions reuse fitted runs and are descriptive, not independent trials.

Candidate age is prediction time minus its first recorded mention, in days.
It is not first-to-last span or condition-era duration. Capped raw sequence
length does not identify the original uncapped history length. Candidate
characterization covers only the predefined stress-eligible subset.

Largest-shift groups use `ceil(0.05 * N)` after ordering absolute calibrated
probability changes at requested 100% repetition; row identifier breaks ties.
Tail and remainder summaries describe different selected groups, not causal
effects or clinical subgroups. No patient or episode rows are published.

Main performance bootstrap: 10,000 patient-cluster draws. Calibration: 2,000
draws. Paper-facing audit bootstrap seed: 20260820. Original fitted models,
calibration fits, and reused evaluation cohort remain fixed during these draws.
Exact commands are in the repository README. The release manifest identifies
public files; it is not a replacement for unavailable source-data provenance.
