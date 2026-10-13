import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Verify public submission files.')
parser.add_argument('--require-source-data', action='store_true', help='Require the separately distributed source archive.')
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]
required = [
    "Project_Report.pdf",
    "PROJECT_REPORT.md",
    "README.md",
    "artifacts/practical/final_practical_model_bundle.joblib",
    "artifacts/practical/final_practical_lightgbm_booster.txt",
    "artifacts/shared/split_indices.csv",
    "artifacts/shared/reproducibility_weight_manifest.json",
    "artifacts/monotonic_lgbm/monotonic_constrained_lgbm_sigmoid.joblib",
    "artifacts/logistic_classweighted/LR_L2_classpow035_C1_sigmoid_calibrated_bundle.joblib",
    "artifacts/logistic_population/LR_L2_population_C1_sigmoid_calibrated_bundle.joblib",
    "outputs/metrics/selected_four_model_comparison.csv",
    "outputs/metrics/selected_four_target_attainment.csv",
    "outputs/metrics/recomputed_selected_four_model_comparison.csv",
    "outputs/metrics/recomputed_target_attainment.csv",
    "outputs/tables/model_role_matrix.csv",
    "outputs/tables/topk_policy_by_model.csv",
    "outputs/tables/score_tier_performance_by_model.csv",
    "outputs/tables/recomputed_topk_policy_by_model.csv",
    "outputs/tables/segment_profile_summary.csv",
    "outputs/tables/segment_top_categories.csv",
    "outputs/tables/optional_cluster_profile_summary.csv",
    "outputs/tables/calibration_bins_by_model.csv",
    "outputs/tables/weighted_confusion_by_model.csv",
    "outputs/tables/practical_lgbm_feature_importance.csv",
    "outputs/figures/role_topk_precision_by_model.png",
    "outputs/figures/role_topk_lift_by_model.png",
    "outputs/figures/practical_lgbm_feature_importance_top20.png",
    "outputs/figures/segment_positive_rate_lift.png",
    "outputs/figures/segment_profile_heatmap.png",
    "docs/segmentation_appendix.md",
    "docs/submission_validation.md",
    "scripts/evaluate_from_artifacts.py",
    "scripts/generate_segmentation_profiles.py",
    "scripts/generate_practical_feature_importance.py",
    "scripts/predict_selected_model.py",
    "scripts/run_full_retraining_pipeline.py",
]
if args.require_source_data:
    required.append("data/TakeHomeProject.zip")
missing = [path for path in required if not (ROOT / path).exists()]
if missing:
    raise SystemExit("Missing required submission files:\n" + "\n".join(missing))
print(f"Public artifact verification passed: {len(required)} files found")
if not args.require_source_data:
    print("Original training data is external. No data-dependent evaluation was performed.")
