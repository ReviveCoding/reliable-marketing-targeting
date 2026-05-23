from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_selected_four_metrics_exist():
    path = ROOT / "outputs" / "metrics" / "selected_four_model_comparison.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) == 4
    assert "REFERENCE_practical_LightGBM" in set(df["model"])


def test_client_report_and_submission_outputs_exist():
    required = [
        ROOT / "Project_Report.pdf",
        ROOT / "PROJECT_REPORT.md",
        ROOT / "README.md",
        ROOT / "docs" / "segmentation_appendix.md",
        ROOT / "outputs" / "tables" / "model_role_matrix.csv",
        ROOT / "outputs" / "tables" / "topk_policy_by_model.csv",
        ROOT / "outputs" / "tables" / "segment_profile_summary.csv",
        ROOT / "outputs" / "figures" / "role_topk_precision_by_model.png",
        ROOT / "outputs" / "figures" / "segment_positive_rate_lift.png",
    ]
    for path in required:
        assert path.exists(), path


def test_model_role_matrix_has_four_roles():
    df = pd.read_csv(ROOT / "outputs" / "tables" / "model_role_matrix.csv")
    assert df["primary_role"].nunique() == 4


def test_optimized_package_has_no_large_redundant_predictions():
    pred_dir = ROOT / "outputs" / "predictions"
    csv_files = list(pred_dir.glob("*.csv")) if pred_dir.exists() else []
    assert not csv_files, "Predictions should be generated on demand, not committed as large redundant CSVs."
