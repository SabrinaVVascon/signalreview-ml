
from pathlib import Path

import pandas as pd


SOURCE = Path("artifacts/erros_teste.csv")

OUTPUT_REVIEW = Path(
    "artifacts/auditoria_100_preencher.csv"
)

OUTPUT_REFERENCE = Path(
    "artifacts/auditoria_100_referencia.csv"
)

SAMPLE_SIZE = 100
RANDOM_STATE = 42


def main():
    print("=== SIGNALREVIEW ML ===")
    print("Criando amostra para auditoria humana...\n")

    if not SOURCE.exists():
        raise FileNotFoundError(
            "Execute primeiro: python -m ml.analyze_errors"
        )

    # Evita sobrescrever uma auditoria ja iniciada.
    if OUTPUT_REVIEW.exists() or OUTPUT_REFERENCE.exists():
        raise FileExistsError(
            "Arquivos de auditoria ja existem. "
            "Verifique-os antes de gerar outra amostra."
        )

    df = pd.read_csv(
        SOURCE,
        encoding="utf-8-sig"
    )

    required_columns = [
        "texto",
        "sentimento_real",
        "sentimento_previsto",
        "confianca",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Colunas ausentes: {missing}"
        )

    # Mantem apenas divergencias entre modelo e estrelas.
    df = df.dropna(
        subset=required_columns
    )

    df = df[
        df["sentimento_real"]
        != df["sentimento_previsto"]
    ].copy()

    n = min(SAMPLE_SIZE, len(df))

    if n == 0:
        raise ValueError(
            "Nenhuma divergencia disponivel."
        )

    # Amostragem aleatoria reproduzivel.
    sample = df.sample(
        n=n,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    sample.insert(
        0,
        "id",
        [f"SR-{i:03d}" for i in range(1, n + 1)]
    )

    # Planilha para preenchimento humano.
    # Nao mostra classificacoes anteriores.
    review = sample[["id", "texto"]].copy()

    review["rotulo_humano"] = ""
    review["justificativa"] = ""

    # Arquivo separado para comparar depois.
    reference = sample[
        [
            "id",
            "sentimento_real",
            "sentimento_previsto",
            "confianca",
        ]
    ].copy()

    OUTPUT_REVIEW.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    review.to_csv(
        OUTPUT_REVIEW,
        index=False,
        sep=";",
        encoding="utf-8-sig"
    )

    reference.to_csv(
        OUTPUT_REFERENCE,
        index=False,
        sep=";",
        encoding="utf-8-sig"
    )

    print(f"Total de divergencias: {len(df)}")
    print(f"Avaliacoes selecionadas: {n}")
    print("\nArquivos gerados:")
    print(OUTPUT_REVIEW)
    print(OUTPUT_REFERENCE)

    print("\nAmostra pronta para classificacao humana!")


if __name__ == "__main__":
    main()
