# Experiments

## Protocol

The study uses EHRSHOT readmission and ICU-transfer tasks with the official
patient-disjoint train, tuning, and held-out splits. The persistent-diagnosis
list, vocabulary, and numeric statistics are derived from train. Each model
uses seeds 42-46; Platt calibration is fitted on tuning predictions.

The training boundary is `event_time <= prediction_time`. The repetition
analysis uses the strictly pre-prediction eligible cohort. Model architectures
are RETAIN-lite with numeric features for readmission and a two-layer numeric
GRU for ICU transfer. The six files in `configs/` define 14 task-representation
pairs, with event budgets of 4,096 and 16,384 and era horizons of 30, 90, and
180 days.

## Installation and Inputs

Training uses Python 3.12; aggregate plotting uses `requirements-public.txt`
in a separate Python 3.13 environment.

```bash
python -m venv .venv-training
source .venv-training/bin/activate
python -m pip install -r requirements.txt
```

Obtain EHRSHOT MEDS and official subject splits through authorized benchmark
access. Place the authorized inputs under `EHRSHOT_MEDS/` or change the paths
in a local copy of the dataset configuration. Keep all restricted inputs,
predictions, caches, and checkpoints outside version control. Start with fresh
caches and output directories when data or configuration changes.

## Local Data Preparation

The published dataset configuration runs locally with tracking and uploads
disabled. Set its data and output paths, then run from the repository root:

```bash
python final_exps/00_build_train_only_persistent_whitelist.py \
  --ehrshot-root EHRSHOT_MEDS \
  --output-dir ehrshot_train_only_chronic_whitelist_50 --skip-upload
python final_exps/01_build_sequence_datasets.py \
  --run-config configs/state_or_space_sequence_datasets.json \
  --notebook-root . --skip-upload --rebuild --rebuild-cache
```

The fixed persistence rule yielded 239 codes on the study inputs;
`--expected-code-count 239` can be used to check that reference cohort.

## Training and Evaluation

Train each configuration locally. The first three use seeds 42-44; the fourth
adds seeds 45-46 to every final task-representation pair. The output directory
names match those expected by the analysis stage:

```bash
for pair in \
  state_or_space_core_4096_runs:core_4096_wide \
  state_or_space_context_16384_runs:context_16384_wide \
  state_or_space_icu_gap_extra_runs:icu_gap_extra_30_180_wide \
  state_or_space_additional_seeds_45_46_runs:additional_seeds_45_46_all_wide
do
  python final_exps/02_train_sequence_multiseed.py \
    --run-config "configs/${pair%%:*}.json" \
    --sequence-data-dir ehrshot_state_or_space_sequence_datasets \
    --output-dir "ehrshot_state_or_space_final_sequence_results/${pair#*:}" \
    --checkpoint-dir checkpoints --skip-checkpoint-upload \
    --results-s3-prefix "" --device auto
done
python final_exps/03_analyze_state_or_space.py \
  --skip-upload --skip-combined-upload
```

Each stage's `--help` lists data, device, checkpoint, and output options. The repetition
stages are `final_exps/04_artificial_copy_forward_inference.py` and
`final_exps/05_analyze_copy_forward_robustness.py`. These stages use frozen
models and do not retrain them.

## Paper Analyses

Given the authorized derived prediction package:

```bash
python paper/analysis/tae_submission_analysis.py \
  --package-root /authorized/package --output-dir /local/results \
  --calibration-bootstrap 2000 --bootstrap-seed 20260820
python paper/analysis/characterize_selected_episodes.py \
  --package-root /authorized/package --output-dir /local/results
python paper/analysis/matched_stability_and_shift_profiles.py \
  --package-root /authorized/package --output-dir /local/results
```

The performance bootstrap uses 10,000 patient-cluster draws; calibration uses
2,000. Seeds change initialization on the same training sample and split.
The published analysis conditions on fitted pipelines and a cohort reused
during development. See the paper for the complete experimental protocol.
