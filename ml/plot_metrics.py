"""Gera graficos reproduziveis do SignalReview ML a partir das metricas salvas.

Execute na raiz do projeto: python -m ml.plot_metrics
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"


def main():
    metrics_file = ARTIFACTS / "metrics.json"
    audit_file = ARTIFACTS / "auditoria_resumo.json"

    if not metrics_file.exists() or not audit_file.exists():
        raise FileNotFoundError(
            "Execute primeiro 'python -m ml.train' e 'python -m ml.analyze_auto'."
        )

    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    audit = json.loads(audit_file.read_text(encoding="utf-8"))

    if metrics.get("demo_data"):
        raise ValueError("Os graficos de portfolio exigem treino com dataset real.")

    FIGURES.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 10, "figure.facecolor": "white"})

    # GRAFICO 1: Comparacao dos modelos na validacao
    names = {
        "dummy_majority": "Baseline",
        "naive_bayes": "Naive Bayes",
        "logistic_regression": "Regressao Logistica",
    }

    models = metrics["validation_models"]
    keys = list(models)

    values = [models[k]["f1_macro"] * 100 for k in keys]

    labels = [names.get(k, k) for k in keys]

    fig, ax = plt.subplots(figsize=(9, 4.8))

    bars = ax.barh(labels, values, color=["#94a3b8", "#60a5fa", "#0f766e"][: len(keys)])

    ax.set_xlim(0, 105)
    ax.set_xlabel("F1-score macro (%)")
    ax.set_title("Comparacao de modelos na validacao")
    ax.invert_yaxis()

    for bar, value in zip(bars, values):
        ax.text(
            value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center"
        )

    fig.tight_layout()

    fig.savefig(FIGURES / "comparacao_modelos.png", dpi=180)

    plt.close(fig)

    # GRAFICO 2: Matriz de confusao
    cm = np.asarray(metrics["confusion_matrix"], dtype=int)

    if cm.shape != (2, 2):
        raise ValueError("Esperada matriz de confusao binaria 2x2.")

    classes = ["Negativo", "Positivo"]

    fig, ax = plt.subplots(figsize=(6.3, 5.4))

    im = ax.imshow(cm, cmap="Blues")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Quantidade")

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Previsao do modelo",
        ylabel="Rotulo das estrelas",
        title="Matriz de confusao - teste",
    )

    threshold = cm.max() / 2

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}".replace(",", "."),
                ha="center",
                va="center",
                fontsize=14,
                color=("white" if cm[i, j] > threshold else "#111827"),
            )

    fig.tight_layout()

    fig.savefig(FIGURES / "matriz_confusao.png", dpi=180)

    plt.close(fig)

    # GRAFICO 3: Auditoria automatica
    groups = audit.get("grupos_de_triagem", {})

    ordered = [
        (
            "possivel_inconsistencia_estrelas",
            "Modelos concordam\ncom estrelas divergentes",
        ),
        ("modelos_divergentes", "Modelos divergem\nentre si"),
        (
            "possivel_inconsistencia_estrelas_alta_pontuacao",
            "Modelos concordam\n(pontuacoes >= 90%)",
        ),
    ]

    labels = [label for key, label in ordered]

    counts = [int(groups.get(key, 0)) for key, _ in ordered]

    fig, ax = plt.subplots(figsize=(9, 5.2))

    bars = ax.barh(labels, counts, color=["#60a5fa", "#94a3b8", "#0f766e"])

    ax.invert_yaxis()

    ax.set_xlim(0, max(counts + [1]) * 1.17)

    ax.set_xlabel("Quantidade de divergencias")

    ax.set_title("Triagem automatica - nao confirma erro nos rotulos")

    for bar, value in zip(bars, counts):
        ax.text(
            value + max(counts + [1]) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
        )

    fig.tight_layout()

    fig.savefig(FIGURES / "triagem_automatica.png", dpi=180)

    plt.close(fig)

    print("Graficos gerados com sucesso:")

    for file in sorted(FIGURES.glob("*.png")):
        print("-", file.relative_to(ROOT))

    print(
        "Aviso: as estrelas sao rotulos aproximados "
        "e a triagem nao e uma validacao humana."
    )


if __name__ == "__main__":
    main()
