
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/processed/reviews.csv")
MODEL_PATH = Path("artifacts/model.joblib")
OUTPUT_PATH = Path("artifacts/erros_teste.csv")


def main():
    print("Carregando dados e modelo...")

    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df = df.dropna(subset=["texto", "sentimento"]).copy()

    df["texto"] = df["texto"].astype(str).str.strip()
    df = df[
        (df["texto"].str.len() >= 3)
        & df["sentimento"].isin(["positivo", "negativo"])
    ]
    df = df.drop_duplicates(subset=["texto"], keep="first")

    # Reproduz exatamente a separacao usada em ml/train.py.
    x_train, x_tmp, y_train, y_tmp = train_test_split(
        df["texto"],
        df["sentimento"],
        test_size=0.30,
        random_state=42,
        stratify=df["sentimento"],
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp,
        y_tmp,
        test_size=0.50,
        random_state=42,
        stratify=y_tmp,
    )

    model = joblib.load(MODEL_PATH)

    previsoes = model.predict(x_test)
    probabilidades = model.predict_proba(x_test)

    classes = list(model.classes_)

    resultados = pd.DataFrame({
        "texto": x_test.to_numpy(),
        "sentimento_real": y_test.to_numpy(),
        "sentimento_previsto": previsoes,
        "probabilidade_positivo": probabilidades[
            :, classes.index("positivo")
        ],
        "probabilidade_negativo": probabilidades[
            :, classes.index("negativo")
        ],
        "confianca": probabilidades.max(axis=1),
    })

    erros = resultados[
        resultados["sentimento_real"]
        != resultados["sentimento_previsto"]
    ].copy()

    erros = erros.sort_values(
        by="confianca", ascending=False
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    erros.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== ANALISE DE ERROS ===")
    print(f"Avaliacoes testadas: {len(resultados)}")
    print(f"Classificacoes corretas: {len(resultados) - len(erros)}")
    print(f"Classificacoes incorretas: {len(erros)}")

    print("\n=== RELATORIO POR CLASSE ===")
    print(
        classification_report(
            resultados["sentimento_real"],
            resultados["sentimento_previsto"],
            zero_division=0,
        )
    )

    print("\n=== EXEMPLOS DE ERROS ===")
    for _, erro in erros.head(5).iterrows():
        print(f"\nTexto: {erro['texto'][:250]}")
        print(f"Real: {erro['sentimento_real']}")
        print(f"Previsto: {erro['sentimento_previsto']}")
        print(f"Confianca: {erro['confianca']:.2%}")

    print(f"\nArquivo de erros salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
