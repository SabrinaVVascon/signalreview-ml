"""Prepara o B2W-Reviews01 para classificacao binaria de sentimentos.

As classes sao aproximacoes baseadas em estrelas, nao rotulos humanos:
1-2 = negativo; 4-5 = positivo; 3 = excluido.
"""

import argparse
import csv
from pathlib import Path

import pandas as pd

COLUMNS = ["review_text", "overall_rating"]


def prepare(input_file: Path, output_file: Path) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_file}")

    # Descobre o separador pelo cabecalho.
    with input_file.open("r", encoding="utf-8-sig", newline="") as csv_file:
        header = csv_file.readline()

    try:
        delimiter = csv.Sniffer().sniff(header, delimiters=",;").delimiter
    except csv.Error as exc:
        raise ValueError(
            "Nao foi possivel reconhecer o separador do CSV (, ou ;)."
        ) from exc

    header_columns = [
        column.strip() for column in next(csv.reader([header], delimiter=delimiter))
    ]

    missing = [column for column in COLUMNS if column not in header_columns]

    if missing:
        raise ValueError(
            f"Colunas nao encontradas: {missing}. "
            f"Separador detectado: {delimiter!r}. "
            f"Colunas disponiveis: {header_columns}"
        )

    print(f"Separador detectado: {delimiter!r}")

    frame = pd.read_csv(
        input_file,
        sep=delimiter,
        usecols=COLUMNS,
        encoding="utf-8-sig",
        low_memory=False,
    )

    frame["overall_rating"] = pd.to_numeric(frame["overall_rating"], errors="coerce")

    frame = frame[frame["overall_rating"].isin([1, 2, 4, 5])].copy()

    frame = frame.dropna(subset=["review_text"])

    frame["texto"] = frame["review_text"].astype(str).str.strip()

    frame = frame[frame["texto"].str.len() >= 10]

    # Evita duplicatas entre treino e teste.
    frame = frame.drop_duplicates(subset=["texto"], keep="first")

    frame["sentimento"] = frame["overall_rating"].map(
        {1: "negativo", 2: "negativo", 4: "positivo", 5: "positivo"}
    )

    prepared = frame[["texto", "sentimento"]].reset_index(drop=True)

    if prepared.empty:
        raise ValueError("Nenhuma avaliacao valida foi encontrada apos os filtros.")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    prepared.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Dados preparados: {len(prepared)} linhas")
    print(prepared["sentimento"].value_counts().to_string())
    print(f"Salvos em: {output_file}")

    return prepared


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input", type=Path, default=Path("data/raw/B2W-Reviews01.csv")
    )

    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/reviews.csv")
    )

    args = parser.parse_args()

    prepare(args.input, args.output)
