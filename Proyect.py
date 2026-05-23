import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# Configuracion de la pagina
st.set_page_config(
    page_title="Iris Species Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

TEAM = ["William Angulo", "Martin Torres"]
UNIVERSITY = "Universidad de la Costa"
COURSE = "Mineria de Datos"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #2d6a4f;
    margin-bottom: 0;
}
.subtitle {
    font-size: 1rem;
    color: #52796f;
    margin-top: 0.2rem;
}
.team-badge {
    display: inline-block;
    background: #d8f3dc;
    color: #1b4332;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.85rem;
    font-weight: 500;
    margin: 3px;
}
.metric-card {
    background: #f0fdf4;
    border-left: 4px solid #2d6a4f;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #2d6a4f;
}
.metric-label {
    font-size: 0.8rem;
    color: #52796f;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #1b4332;
    border-bottom: 2px solid #d8f3dc;
    padding-bottom: 6px;
    margin-bottom: 16px;
}
.predict-result {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #2d6a4f;
    text-align: center;
    padding: 20px;
    background: #d8f3dc;
    border-radius: 12px;
    margin-top: 12px;
}
.stButton > button {
    background-color: #2d6a4f;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    width: 100%;
}
.stButton > button:hover {
    background-color: #1b4332;
}
</style>
""", unsafe_allow_html=True)


# Carga y preparacion del dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Iris.csv")
    # Eliminar columna Id si existe
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    # Limpiar nombres de especies (quitar prefijo "Iris-")
    df["Species"] = df["Species"].str.replace("Iris-", "", regex=False)
    # Renombrar columnas al formato estandar
    df = df.rename(columns={
        "SepalLengthCm": "sepal length (cm)",
        "SepalWidthCm":  "sepal width (cm)",
        "PetalLengthCm": "petal length (cm)",
        "PetalWidthCm":  "petal width (cm)",
        "Species":       "species"
    })
    return df


@st.cache_resource
def train_model(df):
    FEATURES = ["sepal length (cm)", "sepal width (cm)",
                 "petal length (cm)", "petal width (cm)"]
    X = df[FEATURES]
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall":    recall_score(y_test, y_pred, average="macro"),
        "F1 Score":  f1_score(y_test, y_pred, average="macro"),
    }
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    return model, scaler, metrics, cm, X_test, y_test, y_pred


df = load_data()
model, scaler, metrics, cm, X_test, y_test, y_pred = train_model(df)

FEATURES = ["sepal length (cm)", "sepal width (cm)",
            "petal length (cm)", "petal width (cm)"]
SPECIES   = sorted(df["species"].unique().tolist())
COLORS    = {"setosa": "#2d6a4f", "versicolor": "#52b788", "virginica": "#95d5b2"}


# Encabezado principal
st.markdown('<p class="main-title">Iris Species Classifier</p>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">{UNIVERSITY} · {COURSE}</p>', unsafe_allow_html=True)
team_html = " ".join(f'<span class="team-badge">{n}</span>' for n in TEAM)
st.markdown(team_html, unsafe_allow_html=True)
st.markdown("---")


# Navegacion lateral
with st.sidebar:
    st.markdown("### Navegacion")
    page = st.radio(
        "",
        ["Metricas del Modelo", "Prediccion", "Exploracion de Datos"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Equipo**")
    for name in TEAM:
        st.markdown(f"- {name}")
    st.markdown(f"**Modelo:** Random Forest\n\n**Arboles:** 100\n\n**Test set:** 25%")


# PAGINA 1 — Metricas
if page == "Metricas del Modelo":
    st.markdown('<p class="section-header">Metricas de Evaluacion</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    icons = ["Accuracy", "Precision", "Recall", "F1 Score"]
    for col, (name, val) in zip([col1, col2, col3, col4], metrics.items()):
        with col:
            st.markdown(
                f"""<div class="metric-card">
                <div class="metric-label">{name}</div>
                <div class="metric-value">{val:.2%}</div>
                </div>""",
                unsafe_allow_html=True
            )

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-header">Matriz de Confusion</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=SPECIES, yticklabels=SPECIES, ax=ax,
            linewidths=0.5, linecolor="#d8f3dc"
        )
        ax.set_xlabel("Predicho", fontsize=11)
        ax.set_ylabel("Real", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown('<p class="section-header">Importancia de Variables</p>', unsafe_allow_html=True)
        importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        bars = ax2.barh(
            importances.index, importances.values,
            color=["#95d5b2", "#74c69d", "#52b788", "#2d6a4f"]
        )
        ax2.set_xlabel("Importancia", fontsize=11)
        ax2.spines[["top", "right", "left"]].set_visible(False)
        ax2.tick_params(left=False)
        for bar, val in zip(bars, importances.values):
            ax2.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with st.expander("Reporte completo de clasificacion"):
        report_df = pd.DataFrame(
            classification_report(y_test, y_pred, output_dict=True)
        ).T.round(3)
        st.dataframe(report_df, use_container_width=True)


# PAGINA 2 — Prediccion
elif page == "Prediccion":
    st.markdown('<p class="section-header">Prediccion de Especie</p>', unsafe_allow_html=True)
    st.markdown("Ingresa las medidas de la flor y el modelo predecira la especie.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Medidas en cm")
        sl = st.slider("Longitud del sepalo", 4.0, 8.0, 5.8, 0.1)
        sw = st.slider("Ancho del sepalo",    2.0, 4.5, 3.0, 0.1)
        pl = st.slider("Longitud del petalo", 1.0, 7.0, 3.7, 0.1)
        pw = st.slider("Ancho del petalo",    0.1, 2.5, 1.2, 0.1)
        predict_btn = st.button("Predecir especie")

    with col2:
        if predict_btn:
            sample   = np.array([[sl, sw, pl, pw]])
            sample_s = scaler.transform(sample)
            pred     = model.predict(sample_s)[0]
            proba    = model.predict_proba(sample_s)[0]

            st.markdown(
                f'<div class="predict-result">Iris {pred.capitalize()}</div>',
                unsafe_allow_html=True
            )

            st.markdown("#### Probabilidades por especie")
            prob_df = pd.DataFrame({"Especie": model.classes_, "Probabilidad": proba})
            fig_p = px.bar(
                prob_df, x="Especie", y="Probabilidad",
                color="Especie",
                color_discrete_map=COLORS,
                text=prob_df["Probabilidad"].apply(lambda x: f"{x:.1%}"),
                range_y=[0, 1]
            )
            fig_p.update_traces(textposition="outside")
            fig_p.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig_p, use_container_width=True)

            st.markdown("#### Posicion en el dataset (3D)")
            fig3d = px.scatter_3d(
                df,
                x="sepal length (cm)", y="sepal width (cm)", z="petal length (cm)",
                color="species",
                color_discrete_map=COLORS,
                opacity=0.6
            )
            fig3d.add_trace(go.Scatter3d(
                x=[sl], y=[sw], z=[pl],
                mode="markers",
                marker=dict(size=10, color="red", symbol="diamond",
                            line=dict(color="white", width=2)),
                name="Nueva muestra"
            ))
            fig3d.update_layout(margin=dict(t=10, b=10))
            st.plotly_chart(fig3d, use_container_width=True)

        else:
            st.info("Ajusta los sliders y presiona Predecir especie para ver el resultado.")
            fig3d = px.scatter_3d(
                df,
                x="sepal length (cm)", y="sepal width (cm)", z="petal length (cm)",
                color="species",
                color_discrete_map=COLORS,
                opacity=0.6,
                title="Dataset Iris — Scatter 3D"
            )
            fig3d.update_layout(margin=dict(t=40, b=10))
            st.plotly_chart(fig3d, use_container_width=True)


# PAGINA 3 — Exploracion de datos
elif page == "Exploracion de Datos":
    st.markdown('<p class="section-header">Exploracion del Dataset</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Datos", "Histogramas", "Scatter Matrix", "Boxplots"])

    with tab1:
        st.markdown(f"**{len(df)} muestras · 4 variables · 3 clases**")
        st.dataframe(df, use_container_width=True, height=320)
        col_s = st.columns(3)
        for i, sp in enumerate(SPECIES):
            with col_s[i]:
                sub = df[df["species"] == sp]
                st.markdown(f"**{sp.capitalize()}** — {len(sub)} muestras")
                st.dataframe(sub[FEATURES].describe().round(2), use_container_width=True)

    with tab2:
        feat = st.selectbox("Variable", FEATURES)
        fig_h, ax_h = plt.subplots(figsize=(8, 4))
        for sp in SPECIES:
            sub = df[df["species"] == sp][feat]
            ax_h.hist(sub, bins=15, alpha=0.65, label=sp.capitalize(), color=COLORS[sp])
        ax_h.set_xlabel(feat, fontsize=12)
        ax_h.set_ylabel("Frecuencia", fontsize=12)
        ax_h.legend()
        ax_h.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_h)
        plt.close()

    with tab3:
        fig_sm = px.scatter_matrix(
            df, dimensions=FEATURES, color="species",
            color_discrete_map=COLORS,
            opacity=0.7, height=600
        )
        fig_sm.update_traces(diagonal_visible=False, showupperhalf=False, marker_size=4)
        fig_sm.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig_sm, use_container_width=True)

    with tab4:
        feat_b = st.selectbox("Variable para boxplot", FEATURES, key="box")
        fig_b = px.box(
            df, x="species", y=feat_b, color="species",
            color_discrete_map=COLORS, points="all", notched=True
        )
        fig_b.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig_b, use_container_width=True)


# Pie de pagina
st.markdown("---")
st.markdown(
    f"<center><small>{UNIVERSITY} · {COURSE} · "
    + " · ".join(TEAM)
    + "</small></center>",
    unsafe_allow_html=True
)
