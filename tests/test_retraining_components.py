import numpy as np
import pandas as pd

from reliable_marketing.features import FeatureEncoder
from reliable_marketing.metrics import evaluate_binary_scores, topk_metrics
from reliable_marketing.models import positive_class_power_weights, safe_logit


def synthetic_census_frame(n=80):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "age": rng.integers(18, 75, n),
        "class of worker": rng.choice(["Private", "Self-employed-not incorporated", "Not in universe"], n),
        "detailed industry recode": rng.integers(0, 10, n),
        "detailed occupation recode": rng.integers(0, 10, n),
        "education": rng.choice(["High school graduate", "Bachelors degree", "Masters degree"], n),
        "wage per hour": rng.integers(0, 40, n),
        "enroll in edu inst last wk": rng.choice(["Not in universe", "High school", "College"], n),
        "marital stat": rng.choice(["Never married", "Married-civilian spouse present", "Divorced"], n),
        "major industry code": rng.choice(["Retail trade", "Manufacturing-durable goods", "Not in universe"], n),
        "major occupation code": rng.choice(["Professional specialty", "Adm support including clerical", "Not in universe"], n),
        "race": rng.choice(["White", "Black", "Asian or Pacific Islander"], n),
        "hispanic origin": rng.choice(["All other", "Mexican-American"], n),
        "sex": rng.choice(["Male", "Female"], n),
        "member of a labor union": rng.choice(["Not in universe", "No", "Yes"], n),
        "reason for unemployment": rng.choice(["Not in universe", "Job loser"], n),
        "full or part time employment stat": rng.choice(["Full-time schedules", "Not in labor force", "Children or Armed Forces"], n),
        "capital gains": rng.choice([0, 0, 0, 5000, 10000], n),
        "capital losses": rng.choice([0, 0, 1500], n),
        "dividends from stocks": rng.choice([0, 0, 100, 1000], n),
        "tax filer stat": rng.choice(["Nonfiler", "Joint both under 65", "Single"], n),
        "region of previous residence": rng.choice(["Not in universe", "South", "Northeast"], n),
        "state of previous residence": rng.choice(["Not in universe", "New York", "California"], n),
        "detailed household and family stat": rng.choice(["Householder", "Spouse of householder", "Other relative"], n),
        "detailed household summary in household": rng.choice(["Householder", "Spouse of householder", "Child under 18 never married"], n),
        "migration code-change in msa": rng.choice(["?", "MSA to MSA", "Nonmover"], n),
        "migration code-change in reg": rng.choice(["?", "Same county", "Different state same division"], n),
        "migration code-move within reg": rng.choice(["?", "Same county", "Different county same state"], n),
        "live in this house 1 year ago": rng.choice(["Not in universe under 1 year old", "Yes", "No"], n),
        "migration prev res in sunbelt": rng.choice(["?", "Yes", "No"], n),
        "num persons worked for employer": rng.integers(0, 7, n),
        "family members under 18": rng.choice(["Not in universe", "Both parents present", "Mother only present"], n),
        "country of birth father": rng.choice(["United-States", "Mexico", "?"], n),
        "country of birth mother": rng.choice(["United-States", "Mexico", "?"], n),
        "country of birth self": rng.choice(["United-States", "Mexico", "?"], n),
        "citizenship": rng.choice(["Native- Born in the United States", "Foreign born- Not a citizen of U S"], n),
        "own business or self employed": rng.integers(0, 3, n),
        "fill inc questionnaire for veteran's admin": rng.choice(["Not in universe", "No", "Yes"], n),
        "veterans benefits": rng.integers(0, 3, n),
        "weeks worked in year": rng.integers(0, 53, n),
        "year": rng.choice([94, 95], n),
    })


def test_feature_encoder_retraining_shape():
    X = synthetic_census_frame(30)
    enc = FeatureEncoder.fit(X, min_count=1)
    Xmat = enc.transform(X)
    assert Xmat.shape[0] == 30
    assert Xmat.shape[1] == len(enc.feature_names)
    assert np.isfinite(Xmat).all()


def test_retraining_weight_and_metric_helpers():
    y = np.array([0, 0, 1, 1])
    w = np.ones(4)
    score = np.array([0.1, 0.2, 0.8, 0.9])
    weighted = positive_class_power_weights(y, w, 0.35)
    assert weighted.shape == (4,)
    assert np.isfinite(safe_logit(score)).all()
    p, r, lift = topk_metrics(y, score, w, 0.5)
    assert p == 1.0 and r == 1.0 and lift >= 1.0
    row = evaluate_binary_scores("m", y, score, w, y, score, w)
    assert row["AP_w"] == 1.0
