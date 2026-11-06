from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "modelo_manutencao_preditiva.joblib"

PRODUCT_TYPES = {
    "L": "L — qualidade baixa",
    "M": "M — qualidade média",
    "H": "H — qualidade alta",
}

# As chaves são os nomes exatos das colunas usadas no treino.
# "range" traz o mínimo e o máximo do dataset; "default" traz a mediana.
NUMERIC_FIELDS = {
    "Air temperature [K]": {
        "label": "Temperatura do ar (K)",
        "range": (295.3, 304.5),
        "default": 300.1,
        "step": 0.1,
    },
    "Process temperature [K]": {
        "label": "Temperatura do processo (K)",
        "range": (305.7, 313.8),
        "default": 310.1,
        "step": 0.1,
    },
    "Rotational speed [rpm]": {
        "label": "Velocidade de rotação (rpm)",
        "range": (1168, 2886),
        "default": 1503,
        "step": 1,
    },
    "Torque [Nm]": {
        "label": "Torque (Nm)",
        "range": (3.8, 76.6),
        "default": 40.1,
        "step": 0.1,
    },
    "Tool wear [min]": {
        "label": "Desgaste da ferramenta (min)",
        "range": (0, 253),
        "default": 108,
        "step": 1,
    },
}


def to_br(value):
    return str(value).replace(".", ",")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def validate(values):
    errors = []
    for column, field in NUMERIC_FIELDS.items():
        low, high = field["range"]
        if not low <= values[column] <= high:
            errors.append(
                f"**{field['label']}:** informe um valor entre "
                f"{to_br(low)} e {to_br(high)}."
            )
    return errors


st.set_page_config(page_title="Manutenção Preditiva", page_icon="🛠️")

with st.sidebar:
    st.header("Sobre o modelo")
    st.write(
        "Random Forest treinado com o dataset AI4I 2020 "
        "(10.000 registros, 3,39% com falha)."
    )
    st.markdown(
        """
| Métrica (teste) | Resultado |
|:---|---:|
| Acurácia | 98,15% |
| Precisão | 89,74% |
| Recall | 51,47% |
| F1-Score | 65,42% |
| ROC-AUC | 97,09% |
"""
    )
    st.caption(
        "O modelo identificou cerca de metade das falhas reais do conjunto "
        "de teste. Um resultado \"sem falha\" não garante que ela não ocorrerá."
    )

st.title("Predição de Falhas em Máquinas")
st.write(
    "Informe as características operacionais da máquina para estimar "
    "a ocorrência de falha."
)

try:
    model = load_model()
except Exception as error:
    st.error(
        "Não foi possível carregar o modelo. Confira se o arquivo "
        f"`{MODEL_PATH.name}` está junto do `app.py` e se o scikit-learn "
        "instalado é a versão 1.6.1, usada no treino."
    )
    st.caption(f"Detalhe técnico: {error}")
    st.stop()

with st.form("prediction_form"):
    product_type = st.selectbox(
        "Tipo do produto",
        options=list(PRODUCT_TYPES),
        format_func=PRODUCT_TYPES.get,
    )

    values = {}
    columns = st.columns(2)
    for index, (column, field) in enumerate(NUMERIC_FIELDS.items()):
        low, high = field["range"]
        # Sem min_value/max_value: o widget corrigiria o valor em silêncio,
        # e a validação abaixo nunca poderia avisar o usuário.
        values[column] = columns[index % 2].number_input(
            field["label"],
            value=field["default"],
            step=field["step"],
            format="%.1f" if isinstance(field["step"], float) else "%d",
            help=f"Faixa do dataset de treino: {to_br(low)} a {to_br(high)}.",
        )

    submitted = st.form_submit_button("Prever falha", type="primary")

if submitted:
    errors = validate(values)

    if errors:
        st.error(
            "Não foi possível fazer a predição. O modelo só conhece as faixas "
            "do dataset de treino, então corrija os campos abaixo:\n\n"
            + "\n".join(f"- {error}" for error in errors)
        )
    else:
        features = pd.DataFrame([{"Type": product_type, **values}])

        try:
            prediction = int(model.predict(features)[0])
            probability = float(model.predict_proba(features)[0, 1])
        except Exception as error:
            st.error("Ocorreu um erro ao calcular a predição. Tente novamente.")
            st.caption(f"Detalhe técnico: {error}")
            st.stop()

        if prediction == 1:
            st.error("**Classe prevista: Com falha**")
        else:
            st.success("**Classe prevista: Sem falha**")

        st.metric(
            "Probabilidade estimada de falha",
            to_br(f"{probability:.1%}"),
        )
        st.progress(probability)
        st.caption("A classe prevista é \"Com falha\" quando a probabilidade passa de 50%.")
