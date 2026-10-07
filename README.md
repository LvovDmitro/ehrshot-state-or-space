# Similar Scores, Different High-Risk Episodes

**Auditing Model Selection in Longitudinal EHRs**

Polina Korobeinikova, Dmitry Lvov, Ilya Pershin.
Accepted for an in-person poster at TAE (Trust-AI-Eval), NeurIPS 2026.
[Paper and decision](https://openreview.net/forum?id=RffArnWpec).

This maintained fork preserves Polina's original implementation and commit
history. Dmitry Lvov maintains the paper-facing analyses and documentation.
[Original repository](https://github.com/poinka/ehrshot-state-or-space).

## What the study establishes

The study compares raw and persistence-aware EHR sequences for 30-day
readmission and ICU transfer on EHRSHOT. Fourteen task-representation pairs
use five seeds each. Similar aggregate performance coexists with different
top-decile episode sets: 34 of 219 readmission selections and 51 of 204 ICU
selections differ between the primary five-seed ensembles. This is a
fixed-capacity episode comparison, not evidence of clinical benefit or harm.

All ten primary paired intervals include zero. State encoding, retained
history, calibration, weighting, initialization, and controlled repetition are
audited separately. The cohort was reused during development; the findings
are retrospective and do not establish universal compression superiority.
The legacy CSV field `auprc` denotes **average precision (AP)**.

## Offline public reproduction

No EHR records, GPU, MinIO, ClearML, or credentials are needed to verify the
published aggregate arithmetic and regenerate the three bitmap figures.
Tested with Python 3.13; the original training target is Python 3.12.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-public.txt
python paper/reproduce_public.py
python -m pytest -q
```

Outputs go to `paper/reproduced/`. The command verifies release checksums,
primary comparison intervals, selected-set arithmetic, exact strict stress
cohorts, and matched-size stability summaries before plotting.

This is **aggregate verification**, not retraining or a new patient bootstrap.
Those require the authorized restricted inputs described below. The paper's
representation diagram is in LaTeX, not a model output plot.

## Paper materials

- `paper/source/`: final LaTeX, figures, bibliography, official unmodified style.
- `paper/TAE_2026_camera_ready.pdf`: prepared revision; upload status is not implied.
- `paper/aggregate_outputs/`: current camera-ready numerical inputs.
- `paper/analysis/`: audit, calibration, set-stability, stress, and plotting code.
- `paper/PAPER_OUTPUT_MAP.md`: paper-to-file mapping and exact analysis settings.
- `paper/BUILD.md`: PDF build instructions and verified checksum.
- `tests/`: synthetic validation; no clinical records.
- `RELEASE_MANIFEST.csv`: checksums of release files, not historical experiment identity.
- `LICENSING.md`: explicit rights status; no retroactive blanket license.
- `public_results/`: historical July exports, **not** current paper stress results.

The strictly pre-prediction stress cohort has 1,286 readmission and 1,028 ICU
episodes; historical July tables used 1,455 and 1,265. Current analyses never
mix these denominators.

## Authorized row-level reproduction

Obtain EHRSHOT access from its data owner and comply with its Credentialed
Health Data License. Do not upload source records, predictions, identifiers,
checkpoints, or service credentials to this repository.

With the retained private reproducibility package available locally:

```bash
python paper/analysis/tae_submission_analysis.py \
  --package-root /authorized/local/package --output-dir /local/audit \
  --calibration-bootstrap 2000 --bootstrap-seed 20260820
python paper/analysis/characterize_selected_episodes.py \
  --package-root /authorized/local/package --output-dir /local/camera-ready
python paper/analysis/matched_stability_and_shift_profiles.py \
  --package-root /authorized/local/package --output-dir /local/camera-ready
```

These programs emit aggregate outputs. Full training remains in `final_exps/`
with frozen `configs/` and stage wrappers in `scripts/`. Use Python 3.12 and
`requirements.txt` for that historical dependency target; specify local data,
fresh cache/output directories, and your own infrastructure settings. Copy
`env.example` to `.env` only if the optional storage/tracking pipeline is used.
`storage.invalid` is an intentional non-routable placeholder, not a server.

The trained input boundary was `event_time <= prediction_time`; the separate
repetition audit excludes episodes with eligible visits at or after prediction.
Do not silently change the trained boundary and report the old results.

Important limits: the exact upstream MEDS archive revision/checksum and training
hardware were not retained; source records cannot be redistributed. Historical
cache/resume checks are not content-addressed. Start with fresh caches/runs when
changing data, whitelist, vocabulary, or configuration. The historical command
reference is `docs/TRAINING_PIPELINE_HISTORY.md`; current scripts supersede it.

## Attribution and support

Innopolis University is the first affiliation for all authors. Dmitry Lvov is
also affiliated with ITMO University; Ilya Pershin with Kazan Federal University.

The study was supported by the Ministry of Economic Development of the Russian
Federation (agreement No. 139-10-2025-034 dd. 19.06.2025,
IGK 000000C313925P4D0002).

Use `CITATION.cff` for attribution. See `LICENSING.md` before reusing code;
public availability does not grant unrestricted reuse rights.
