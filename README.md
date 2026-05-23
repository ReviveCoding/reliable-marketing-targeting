# Census Income Marketing Targeting Take-Home

Recommended default model: **`practical` = `REFERENCE_practical_LightGBM`**.

This package contains a concise client-facing retail marketing solution for classifying likely $50K+ individuals and converting calibrated model scores into actionable marketing segments. The final submission is organized around one recommended model, one constrained challenger, two transparent benchmarks, score-tier profile segmentation, and reproducible evaluation from saved artifacts.

## Recommended review order

1. `Project_Report.pdf` or `Project_Report.docx` - concise client-facing report formatted for page-limited review.
2. `PROJECT_REPORT.md` - editable Markdown source for the client-facing report.
3. `README.md` - setup, run, evaluation, segmentation, prediction, and retraining instructions.
4. `scripts/evaluate_from_artifacts.py` - recomputes held-out metrics from submitted artifacts.
5. `scripts/generate_segmentation_profiles.py` - creates score-tier profile segmentation and optional clustering appendix.
6. `scripts/predict_selected_model.py` - generates full-dataset scores and score tiers.
7. `scripts/generate_practical_feature_importance.py` - exports the final model's feature-importance table and plot.
8. `scripts/run_full_retraining_pipeline.py` - retrains selected model families from raw data.
9. `docs/segmentation_appendix.md` - optional details for score-tier profiles and independent clustering.

## Final model comparison

The table below is recomputed from the submitted artifacts on the held-out test split.

| Model | AP | PR-AUC trapz | ROC-AUC | F1 | Brier | ECE10 | Precision@Top5 | Lift@Top5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `REFERENCE_practical_LightGBM` | 0.6953 | 0.6952 | 0.9550 | 0.6341 | 0.0327 | 0.0024 | 0.6995 | 10.83x |
| `Monotonic_constrained_LightGBM` | 0.6869 | 0.6868 | 0.9537 | 0.6258 | 0.0339 | 0.0060 | 0.6864 | 10.63x |
| `LR_L2_classpow035_C1` | 0.6143 | 0.6145 | 0.9450 | 0.5792 | 0.0369 | 0.0044 | 0.6267 | 9.71x |
| `LR_L2_population_C1` | 0.6237 | 0.6236 | 0.9458 | 0.5757 | 0.0365 | 0.0035 | 0.6362 | 9.85x |

AP is the primary precision-recall summary metric. `PR_AUC_trapz_w` is included as a supplementary PR-AUC calculation for reviewer convenience.

## Model roles

| Model key | Model | Role | When to use |
|---|---|---|---|
| `practical` | `REFERENCE_practical_LightGBM` | Final recommended model | Default scoring, final report, calibrated campaign targeting |
| `monotonic` | `Monotonic_constrained_LightGBM` | Business-constrained challenger | Use when directional business logic and model-risk review matter |
| `logistic_classweighted` | `LR_L2_classpow035_C1` | Interpretable class-weighted benchmark | Use for coefficient-level explanation with positive-class emphasis |
| `logistic_population` | `LR_L2_population_C1` | Transparent population-weighted baseline | Use as the simplest weighted linear reference model |

Artifact organization follows the four retained model roles above: each retained model now has its own artifact folder.

## Project layout

```text
Project_Report.pdf              Page-limited client-facing report
Project_Report.docx             Editable Word version of the client-facing report
PROJECT_REPORT.md               Editable Markdown source for the client-facing report
README.md                       Reviewer entry point and run instructions
artifacts/                      Saved model bundles plus shared split/reproducibility metadata
  practical/                    Final recommended LightGBM model
  monotonic_lgbm/               Business-constrained LightGBM challenger
  logistic_classweighted/       Interpretable class-weighted benchmark
  logistic_population/          Transparent population-weighted baseline
data/TakeHomeProject.zip         Original take-home data archive
scripts/                        Evaluation, segmentation, prediction, retraining, and verification entrypoints
src/reliable_marketing/          Reusable data, feature, metric, model, and segmentation code
outputs/metrics/                Final and recomputed model metrics
outputs/tables/                 Top-k, segmentation, calibration, constraint, and model-role tables
outputs/figures/                Core model and segmentation visualizations
docs/segmentation_appendix.md    Optional segmentation details and clustering appendix
tests/                          Lightweight smoke tests
```

The package includes only `data/TakeHomeProject.zip` for the raw take-home data. Scripts automatically extract it when needed, so the extracted raw-data folder is intentionally not included in the submitted ZIP.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Recompute final evaluation metrics

```bash
python scripts/evaluate_from_artifacts.py --models all
```

This reloads the submitted artifacts, reconstructs the calibration/test split, recomputes weighted classification, PR, calibration, and Top-K marketing metrics, and writes:

```text
outputs/metrics/recomputed_selected_four_model_comparison.csv
outputs/metrics/recomputed_target_attainment.csv
outputs/metrics/recomputed_vs_stored_metric_deltas.csv
outputs/tables/recomputed_topk_policy_by_model.csv
outputs/tables/calibration_bins_by_model.csv
outputs/tables/weighted_confusion_by_model.csv
```

`python scripts/evaluate_selected_four_models.py` is kept as a compatibility wrapper around the same artifact-based evaluation path.

## Generate segmentation profiles

Main score-tier profile segmentation:

```bash
python scripts/generate_segmentation_profiles.py --model practical --split test
```

Main segmentation plus optional independent clustering appendix:

```bash
python scripts/generate_segmentation_profiles.py --model practical --split test --include-clusters --k-values 4
```

Outputs:

```text
outputs/tables/segment_profile_summary.csv
outputs/tables/segment_top_categories.csv
outputs/tables/optional_cluster_profile_summary.csv
outputs/tables/optional_cluster_silhouette_scores.csv  # includes inertia and silhouette diagnostics
docs/segmentation_appendix.md
outputs/figures/segment_positive_rate_lift.png
outputs/figures/segment_profile_heatmap.png
```

The score-tier profile segmentation is the recommended activation layer. The optional clustering output excludes the target and classifier score and is provided only as a profile-discovery appendix.

## Generate predictions

```bash
python scripts/predict_selected_model.py --model practical --include-labels
```

By default, predictions are written to `outputs/predictions/<model>_full_scores.csv`. Large prediction CSVs are generated on demand rather than committed to keep the repository lightweight.

## Generate final-model explanation artifacts

```bash
python scripts/generate_practical_feature_importance.py
```

This writes:

```text
outputs/tables/practical_lgbm_feature_importance.csv
outputs/figures/practical_lgbm_feature_importance_top20.png
```

These artifacts explain the final recommended model without changing the frozen submitted weights.

## Verify package artifacts

```bash
python scripts/verify_selected_four_artifacts.py
pytest -q
```

## GitHub/local smoke test

The included GitHub Actions workflow mirrors the local smoke-test path below and does not call any scripts that are absent from the package.

```bash
python scripts/verify_selected_four_artifacts.py
python scripts/evaluate_selected_four_models.py
pytest -q
python scripts/run_full_retraining_pipeline.py --models practical --quick-rows 200 --n-estimators 1 --out-dir /tmp/retraining_smoke
python scripts/predict_selected_model.py --model practical --include-labels --out /tmp/practical_ci_scores.csv
```

## Retrain selected model families from raw data

The submitted best model weights are frozen under `artifacts/`. Retraining is intentionally saved to a separate run directory so the official submission artifacts are not overwritten. By default, each retraining command creates a timestamped folder under `retraining_runs/`:

```bash
python scripts/run_full_retraining_pipeline.py --models all
```

Example output folder:

```text
retraining_runs/run_YYYYMMDD_HHMMSS_full_all/
  artifacts/
  metrics/
  run_summary.json
```

Fast smoke retraining pass:

```bash
python scripts/run_full_retraining_pipeline.py \
  --models practical \
  --quick-rows 200 \
  --n-estimators 1
```

Optional named run:

```bash
python scripts/run_full_retraining_pipeline.py \
  --models practical \
  --run-name practical_smoke_check
```

If `--out-dir` is provided and that directory already exists, the script automatically writes to a suffixed folder such as `_001`, `_002`, and so on. The script refuses to write inside `artifacts/`, because `artifacts/` contains the official submitted model weights.

## Notes for reviewers

- Population/sample weight is used as a weight, not as a model feature.
- Ranking, classification, and probability-calibration metrics are reported separately.
- The final segmentation is a score-tier profile segmentation, not just a raw score bucket.
- Logistic regression models are transparent benchmarks, not the final recommended scoring model.
- The monotonic model constrains selected business-reasonable numeric features only; it does not force every categorical effect to be monotonic.
- `.joblib` model artifacts are included for reviewer convenience and should be loaded only in a trusted local environment. Full retraining from raw data is supported.
