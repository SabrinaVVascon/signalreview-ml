import pandas as pd
from ml.train import train


def test_train_creates_model_and_metrics(tmp_path):
    src = pd.read_csv("data/demo_reviews.csv")
    file_path = tmp_path / "data.csv"
    src.to_csv(file_path, index=False)
    report = train(file_path, tmp_path / "artifacts", is_demo=True)
    assert (tmp_path / "artifacts" / "model.joblib").exists()
    assert (tmp_path / "artifacts" / "metrics.json").exists()
    assert set(report["validation_models"]) == {"dummy_majority", "naive_bayes", "logistic_regression"}
    assert 0 <= report["test_metrics"]["f1_macro"] <= 1
