
"""Interface publica do SignalReview ML com Streamlit."""

import joblib
import streamlit as st
from huggingface_hub import hf_hub_download


MODEL_REPO = "SabrinaVV/signalreview-ml"


st.set_page_config(
    page_title="SignalReview ML",
    page_icon="🧠",
    layout="centered",
)


@st.cache_resource
def carregar_modelo():
    caminho = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="model.joblib",
    )

    # Modelo publicado pela propria autora.
    return joblib.load(caminho)


st.title("🧠 SignalReview ML")

st.write(
    "Classificação de sentimentos em português "
    "utilizando Machine Learning e NLP."
)

st.caption(
    "TF-IDF + Regressão Logística | "
    "110.882 avaliações utilizadas"
)

st.divider()

st.subheader("Analisar avaliação")

texto = st.text_area(
    "Digite uma avaliação de produto:",
    placeholder=(
        "Exemplo: O produto chegou rápido, "
        "excelente qualidade e ótimo atendimento!"
    ),
    height=140,
    max_chars=3000,
)

if st.button(
    "Analisar sentimento",
    type="primary",
    use_container_width=True,
):
    if not texto.strip():
        st.warning("Digite uma avaliação para analisar.")

    else:
        try:
            with st.spinner("Analisando sentimento..."):
                modelo = carregar_modelo()

                previsao = str(
                    modelo.predict([texto.strip()])[0]
                )

                probabilidades = modelo.predict_proba(
                    [texto.strip()]
                )[0]

                scores = {
                    str(classe): float(score)
                    for classe, score in zip(
                        modelo.classes_,
                        probabilidades,
                    )
                }

            st.divider()
            st.subheader("Resultado da análise")

            if previsao == "positivo":
                st.success("Sentimento identificado: POSITIVO")
            else:
                st.error("Sentimento identificado: NEGATIVO")

            for classe in ["positivo", "negativo"]:
                probabilidade = scores.get(classe, 0.0)

                st.write(
                    f"**{classe.capitalize()}: "
                    f"{probabilidade:.2%}**"
                )

                st.progress(probabilidade)

            st.caption(
                "As pontuações são estimativas do modelo "
                "e não representam probabilidades "
                "calibradas de acerto."
            )

        except Exception:
            st.error(
                "Não foi possível carregar ou executar "
                "o modelo. Verifique os logs da aplicação."
            )
            st.stop()

st.divider()

st.subheader("Sobre o modelo")

col1, col2 = st.columns(2)

with col1:
    st.metric("Acurácia no teste", "94,81%")

with col2:
    st.metric("F1-score macro", "93,92%")

st.caption(
    "Métricas calculadas com rótulos derivados "
    "das estrelas dos consumidores."
)

st.markdown(
    """
    **Tecnologias:** Python, scikit-learn,
    TF-IDF, Regressão Logística e Streamlit.

    **Links do projeto:**

    - [Código-fonte no GitHub](https://github.com/SabrinaVVascon/signalreview-ml)
    - [Modelo no Hugging Face](https://huggingface.co/SabrinaVV/signalreview-ml)

    **Limitações:** classificação binária;
    avaliações irônicas, neutras ou ambíguas
    podem apresentar resultados inadequados.
    """
)
