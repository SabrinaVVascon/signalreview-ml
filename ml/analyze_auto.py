
"""Auditoria automatica de divergencias entre estrelas e classificadores.

Execute na raiz do projeto: python -m ml.analyze_auto
Nao substitui rotulos humanos; identifica casos para investigacao.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from ml.train import build_candidates


DATA = Path("data/processed/reviews.csv")
MODEL = Path("artifacts/model.joblib")
METRICS = Path("artifacts/metrics.json")
OUTPUT = Path("artifacts")
LABELS = {"positivo", "negativo"}


def main():
    if not DATA.exists() or not MODEL.exists() or not METRICS.exists():
        raise FileNotFoundError(
            "Faltam arquivos. Execute primeiro: python -m ml.prepare_data "
            "e depois: python -m ml.train"
        )

    info = json.loads(METRICS.read_text(encoding="utf-8"))
    if info.get("demo_data"):
        raise ValueError("Treine com os dados reais antes da auditoria.")

    # Mesmo tratamento e divisao usados em ml/train.py.
    df = pd.read_csv(DATA, encoding="utf-8")
    df = df.dropna(subset=["texto", "sentimento"]).copy()
    df["texto"] = df["texto"].astype(str).str.strip()
    df = df[
        (df["texto"].str.len() >= 3)
        & df["sentimento"].isin(LABELS)
    ]
    df = df.drop_duplicates(subset=["texto"], keep="first")

    x_train, x_tmp, y_train, y_tmp = train_test_split(
        df["texto"], df["sentimento"],
        test_size=0.30,
        random_state=42,
        stratify=df["sentimento"],
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp,
        test_size=0.50,
        random_state=42,
        stratify=y_tmp,
    )

    modelo_principal = joblib.load(MODEL)
    nome_principal = info.get(
        "selected_model", "modelo_principal"
    )

    pred_principal = modelo_principal.predict(x_test)
    scores_principal = modelo_principal.predict_proba(
        x_test
    ).max(axis=1)

    # Treina o segundo modelo usando somente treino e validacao.
    # Nenhum dos modelos recebe os exemplos de teste no treino.
    x_full = pd.concat([x_train, x_val])
    y_full = pd.concat([y_train, y_val])

    segundo_modelo = build_candidates(
        min_df=2 if len(x_full) >= 100 else 1
    )["naive_bayes"]

    print("Treinando auditor auxiliar: Naive Bayes...")
    segundo_modelo.fit(x_full, y_full)

    pred_auxiliar = segundo_modelo.predict(x_test)
    scores_auxiliar = segundo_modelo.predict_proba(
        x_test
    ).max(axis=1)

    resultados = pd.DataFrame({
        "texto": x_test.to_numpy(),
        "rotulo_estrelas": y_test.to_numpy(),
        "previsto_principal": pred_principal,
        "previsto_auxiliar": pred_auxiliar,
        "score_principal": scores_principal.round(4),
        "score_auxiliar": scores_auxiliar.round(4),
    })

    # Seleciona divergencias entre o modelo principal e estrelas.
    erros = resultados[
        resultados["previsto_principal"]
        != resultados["rotulo_estrelas"]
    ].copy()

    concordam = (
        erros["previsto_principal"]
        == erros["previsto_auxiliar"]
    )

    ambos_altos = (
        (erros["score_principal"] >= 0.90)
        & (erros["score_auxiliar"] >= 0.90)
    )

    # Triagem automatica: categorias de investigacao,
    # nao classificacoes humanas definitivas.
    erros["triagem"] = "modelos_divergentes"

    erros.loc[
        concordam, "triagem"
    ] = "possivel_inconsistencia_estrelas"

    erros.loc[
        concordam & ambos_altos, "triagem"
    ] = "possivel_inconsistencia_estrelas_alta_pontuacao"

    erros = erros.sort_values(
        ["triagem", "score_principal"],
        ascending=[True, False]
    ).reset_index(drop=True)

    resumo = {
        "observacao": (
            "Rotulos derivados de estrelas nao sao verdade humana. "
            "Concordancia de dois modelos nao confirma qual rotulo "
            "esta certo. Scores nao equivalem a probabilidades calibradas."
        ),
        "modelo_principal": nome_principal,
        "modelo_auxiliar": "naive_bayes",
        "total_teste": int(len(resultados)),
        "divergencias_principal_vs_estrelas": int(len(erros)),
        "acuracia_principal_vs_estrelas": round(
            float(accuracy_score(y_test, pred_principal)), 4
        ),
        "f1_macro_principal_vs_estrelas": round(
            float(f1_score(
                y_test, pred_principal, average="macro"
            )), 4
        ),
        "acuracia_auxiliar_vs_estrelas": round(
            float(accuracy_score(y_test, pred_auxiliar)), 4
        ),
        "f1_macro_auxiliar_vs_estrelas": round(
            float(f1_score(
                y_test, pred_auxiliar, average="macro"
            )), 4
        ),
        "grupos_de_triagem": {
            str(k): int(v)
            for k, v in erros["triagem"].value_counts().items()
        },
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT / "auditoria_automatica.csv"
    json_path = OUTPUT / "auditoria_resumo.json"
    md_path = OUTPUT / "relatorio_auditoria.md"

    erros.to_csv(
        csv_path,
        sep=";",
        index=False,
        encoding="utf-8-sig"
    )

    json_path.write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    texto_relatorio = [
        "# SignalReview ML - Auditoria automatica de classificacoes",
        "",
        "## Metodo",
        "Regressao Logistica salva versus Naive Bayes treinado "
        "em treino + validacao.",
        "Ambos avaliados em exemplos de teste nao usados no treino.",
        "Os rotulos de referencia foram derivados das estrelas.",
        "",
        "## Resultados",
        f"- Exemplos de teste: {resumo['total_teste']}",
        "- Divergencias modelo principal vs estrelas: "
        f"{resumo['divergencias_principal_vs_estrelas']}",
        "- Acuracia principal vs estrelas: "
        f"{resumo['acuracia_principal_vs_estrelas']:.2%}",
        "- F1-macro principal vs estrelas: "
        f"{resumo['f1_macro_principal_vs_estrelas']:.2%}",
        "- Acuracia Naive Bayes vs estrelas: "
        f"{resumo['acuracia_auxiliar_vs_estrelas']:.2%}",
        "",
        "## Triagem automatica das divergencias",
    ]

    for grupo, quantidade in resumo["grupos_de_triagem"].items():
        texto_relatorio.append(f"- {grupo}: {quantidade}")

    texto_relatorio += [
        "",
        "## Limitacoes",
        "- Dois modelos podem errar juntos.",
        "- Possivel inconsistencia nao confirma erro nas estrelas.",
        "- Alta pontuacao nao garante classificacao correta.",
        "- O conjunto de teste nao deve ser usado para retreino.",
        "- Verificar licenca e privacidade antes de publicar textos.",
    ]

    md_path.write_text(
        "\n".join(texto_relatorio) + "\n",
        encoding="utf-8"
    )

    print("\n=== AUDITORIA AUTOMATICA CONCLUIDA ===")
    print(f"Avaliacoes de teste: {resumo['total_teste']}")
    print(
        "Divergencias: "
        f"{resumo['divergencias_principal_vs_estrelas']}"
    )

    for grupo, quantidade in resumo["grupos_de_triagem"].items():
        print(f"  {grupo}: {quantidade}")

    print("\nArquivos gerados:")
    print(f"- {csv_path}")
    print(f"- {json_path}")
    print(f"- {md_path}")
    print(
        "\nATENCAO: triagem automatica, "
        "nao rotulagem humana definitiva."
    )


if __name__ == "__main__":
    main()
