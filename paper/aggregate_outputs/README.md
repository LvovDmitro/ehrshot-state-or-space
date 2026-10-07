# Aggregate Results

Numerical inputs for the paper's tables and figures. The `auprc` field denotes
average precision (AP). Results contain no patient or episode identifiers.

| Analysis | Files |
|---|---|
| Ensemble performance and paired contrasts | `recomputed_ensemble_metrics.csv`, `multiplicity_sensitivity.csv` |
| Calibration | `calibration_summary.csv`, `calibration_bins.csv` |
| Protocol-dependent preferences | `evaluation_conclusion_matrix.csv` |
| Selected-set overlap and retraining stability | `ensemble_top10_cross_representation.csv`, `matched_size_stability.csv`, `seed_top10_*.csv` |
| Discordant episode profiles | `discordant_top_decile_characterization.csv` |
| Strict pre-prediction repetition | `copy_forward_strict_pre_prediction_*.csv` |
| Membership exchanges and largest shifts | `full_repetition_top_decile_churn.csv`, `stress_shift_tail_characterization.csv` |

Selection uses `ceil(0.10 * N)` with deterministic tie-breaking. Candidate age
is time since its first recorded mention. Largest-shift profiles use the top
5% of absolute probability changes within the strict stress cohort. The
fixed-size retraining references use one- and two-seed ensembles.

Run `python paper/reproduce_public.py` from the repository root to regenerate
the tables and figures. Recomputing patient-level statistics requires the
authorized inputs described in [the experiment guide](../../docs/EXPERIMENTS.md).
