"""Treina baselines, seleciona via validacao e avalia UMA VEZ no teste.

Exemplo:
    python -m ml.train --data data/demo_reviews.csv --demo
"""
import argparse
import json
from pathlib import Path
import time

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def build_candidates(min_df: int) -> dict:
    def vectorizer():
        return TfidfVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
            min_df=min_df, max_features=20000, sublinear_tf=True,
        )
    return {
        "dummy_majority": Pipeline([("tfidf", vectorizer()), ("classifier", DummyClassifier(strategy="most_frequent"))]),
        "naive_bayes": Pipeline([("tfidf", vectorizer()), ("classifier", MultinomialNB())]),
        "logistic_regression": Pipeline([("tfidf", vectorizer()), ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced"))]),
    }


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def train(data_path: Path, artifacts_dir: Path, is_demo: bool = False) -> dict:
    df = pd.read_csv(data_path, encoding="utf-8")
    required = {"texto", "sentimento"}
    if not required.issubset(df.columns):
        raise ValueError("CSV precisa das colunas 'texto' e 'sentimento'.")
    df = df.dropna(subset=["texto", "sentimento"]).copy()
    df["texto"] = df["texto"].astype(str).str.strip()
    df = df[(df["texto"].str.len() >= 3) & df["sentimento"].isin(["positivo", "negativo"])]
    df = df.drop_duplicates(subset=["texto"], keep="first")
    if len(df) < 30 or df["sentimento"].value_counts().min() < 10:
        raise ValueError("Precisamos de ao menos 30 exemplos distintos e 10 exemplos de cada classe.")
    # Divisao 70% treino, 15% validacao, 15% teste, estratificada.
    x_train, x_tmp, y_train, y_tmp = train_test_split(
        df["texto"], df["sentimento"], test_size=0.30,
        random_state=42, stratify=df["sentimento"],
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp,
    )

    results = {}
    best_name, best_f1, best_model = None, -1, None
    min_df = 1 if len(x_train) < 100 else 2
    for name, candidate in build_candidates(min_df).items():
        start = time.perf_counter()
        candidate.fit(x_train, y_train)
        elapsed = round(time.perf_counter() - start, 3)
        val_pred = candidate.predict(x_val)
        result = metrics(y_val, val_pred)
        results[name] = {**result, "train_seconds": elapsed}
        if result["f1_macro"] > best_f1:
            best_name, best_f1, best_model = name, result["f1_macro"], candidate

    # Reajusta o melhor algoritmo em treino+validacao, sem tocar no teste.
    x_full = pd.concat([x_train, x_val])
    y_full = pd.concat([y_train, y_val])
    best_model.fit(x_full, y_full)
    y_pred = best_model.predict(x_test)
    test_metrics = metrics(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["negativo", "positivo"]).tolist()
    infer_start = time.perf_counter()
    for _ in range(100):
        best_model.predict(["A entrega foi boa e o produto tem qualidade."])
    latency_ms = round((time.perf_counter() - infer_start) * 10, 3)
    report = {
        "dataset": str(data_path),
        "demo_data": is_demo,
        "note": "Metricas de demonstracao nao sao evidencias de desempenho real." if is_demo else "Rotulos proxy gerados de estrelas, nao anotacao humana direta.",
        "samples": {"total": len(df), "train": len(x_train), "validation": len(x_val), "test": len(x_test)},
        "class_counts": df["sentimento"].value_counts().to_dict(),
        "validation_models": results,
        "selected_model": best_name,
        "test_metrics": test_metrics,
        "confusion_matrix_labels": ["negativo", "positivo"],
        "confusion_matrix": cm,
        "inference_latency_ms_mean_100_calls": latency_ms,
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, artifacts_dir / "model.joblib")
    (artifacts_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nModelo salvo:", artifacts_dir / "model.joblib")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/reviews.csv"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--demo", action="store_true", help="Marca metricas como demonstracao; nao usar para portifolio")
    args = parser.parse_args()
    if args.data.name == "demo_reviews.csv" and not args.demo:
        parser.error("Para treino com dados sinteticos, adicione --demo.")
    train(args.data, args.artifacts, is_demo=args.demo)
