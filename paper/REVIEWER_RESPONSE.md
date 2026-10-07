# Camera-ready response to reviewer b25p

The camera-ready version adds descriptive analyses from retained predictions;
no new model training was performed. The reviewed submission was V3. Editorial
corrections from the later V5 draft are incorporated without replacing the
underlying experimental results.

## 1. Reference ordinary model instability

Section 4.4 and Table 4 now compare top-decile Jaccard overlap at matched
ensemble sizes. Within-representation comparisons use disjoint one- or two-seed
subsets; cross-representation comparisons use the same seed-index subset.
Pair counts are 10/15 within and 5/10 across. Comparisons reuse fitted runs and
are descriptive. Five-seed ensemble overlap remains a separate observed result.
The earlier unmatched single-model versus five-model attribution is removed.

These initialization references do not test resampled training patients or
alternative splits; that stronger request remains untested and is disclosed.

## 2. Characterize discordant episodes

Section 4.5 and Table 5 report episode/patient/outcome counts and the available
candidate-diagnosis profiles. Coverage is explicit (41–82%). The previously
misnamed span is corrected to time since first candidate mention; retained
length at the context cap is distinguished from uncapped history length.
The descriptions are conditional on stress eligibility and do not establish
clinical causes or a universal persistence signature.

## 3. Link perturbations to episode-level changes

Section 4.6 and Tables 7–8 now report exact top-decile entries/exits and profiles
of the largest 5% probability shifts. Readmission exchanges involve one episode
per pipeline; ICU exchanges involve three. The maximum ICU Era+backfill shift
is approximately 9.8 percentage points despite small mean shifts. Tail profiles
are selected separately by pipeline; intervention dose increases with eligible
visit count. Neither tail membership nor observed outcomes establish clinical
benefit, causal mechanisms, or the causes of ablation-induced discordance.

All added summaries are provided as publication-safe aggregates with executable
analysis code. The primary paper still discloses adaptive held-out reuse,
conditional bootstrap uncertainty, residual calibration error, and the absence
of external validation.
