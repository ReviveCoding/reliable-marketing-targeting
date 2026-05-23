# Submission Validation Log

This file records the local checks run after the final cleanup.

## Fixes applied

- Removed obsolete GitHub Actions steps that called missing scripts.
- Added `scripts/generate_practical_feature_importance.py`.
- Added final Practical LightGBM explanation artifacts:
  - `outputs/tables/practical_lgbm_feature_importance.csv`
  - `outputs/figures/practical_lgbm_feature_importance_top20.png`
- Updated optional clustering diagnostics so `outputs/tables/optional_cluster_silhouette_scores.csv` now includes both inertia and silhouette diagnostics.
- Regenerated `Project_Report.docx` and `Project_Report.pdf` from the updated Markdown report.
- Verified that `outputs/predictions/practical_full_scores.csv` can be generated for the final recommended model. The CSV is intentionally not committed because it is a large derived artifact.

## Validation commands and results

```bash
PYTHONPATH=src python scripts/verify_selected_four_artifacts.py
# Lean final submission verification passed: 37 files found

PYTHONPATH=src python scripts/evaluate_selected_four_models.py
# Recomputed held-out metrics for all four retained models.
# Practical LightGBM: AP=0.695268, ROC-AUC=0.954991, F1=0.634118,
# Brier=0.032690, ECE10=0.002359, Precision@Top5=0.699499, Lift@Top5=10.833277.

PYTHONPATH=src pytest -q
# 6 passed

PYTHONPATH=src python scripts/run_full_retraining_pipeline.py --models practical --quick-rows 200 --n-estimators 1 --out-dir /tmp/retraining_smoke
# Completed and saved smoke retraining outputs to /tmp/retraining_smoke

PYTHONPATH=src python scripts/predict_selected_model.py --model practical --include-labels --out outputs/predictions/practical_full_scores.csv
# saved outputs/predictions/practical_full_scores.csv (199523, 5); removed afterward because predictions are generated on demand.
```

## Report render QA

`Project_Report.docx` was rendered with LibreOffice through the DOCX render workflow. The regenerated report has 8 pages, within the 10-page limit. The rendered pages were visually inspected for layout, clipping, missing figures, and broken tables.
