from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ModuleNotFoundError:  # pragma: no cover - handled in the UI
    px = None
    go = None

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
RISK_ORDER = ["Low", "Medium", "High"]
RISK_LABELS = {"Low": "Bajo", "Medium": "Medio", "High": "Alto"}


st.set_page_config(
    page_title="ALDIMI Predict",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    health = pd.read_csv(DATA_DIR / "Dataset_ALDIMI_Merged_Clean.csv")
    stock = pd.read_csv(DATA_DIR / "stock_structured.csv", parse_dates=["Fecha"])
    health_daily = pd.read_csv(DATA_DIR / "health_daily.csv", parse_dates=["Fecha"])

    health["Risk_Lifestyle_Score"] = (
        health["Smoking"]
        + health["Alcohol_Use"]
        + health["Obesity"]
        + health["Air_Pollution"]
        + health["Occupational_Hazards"]
    ) / 5
    health["Diet_Risk_Index"] = (
        health["Diet_Red_Meat"] + health["Diet_Salted_Processed"] + (10 - health["Fruit_Veg_Intake"])
    ) / 3
    health["Risk_Level"] = pd.Categorical(health["Risk_Level"], categories=RISK_ORDER, ordered=True)

    stock = stock.sort_values(["ID_Insumo", "Fecha"]).copy()
    stock["Consumo_7d"] = stock.groupby("ID_Insumo")["Consumo_Diario"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    stock["Consumo_14d"] = stock.groupby("ID_Insumo")["Consumo_Diario"].transform(
        lambda s: s.rolling(14, min_periods=1).mean()
    )
    stock["Cobertura_Dias"] = stock["Stock_Actual"] / stock["Consumo_7d"].replace(0, np.nan)
    stock["Stock_Proyectado_7d"] = (stock["Stock_Actual"] - (stock["Consumo_7d"] * 7)).clip(lower=0)
    stock["Stock_Proyectado_14d"] = (stock["Stock_Actual"] - (stock["Consumo_14d"] * 14)).clip(lower=0)
    stock["Alerta"] = np.select(
        [
            stock["Cobertura_Dias"] <= stock["Lead_Time"],
            stock["Cobertura_Dias"] <= stock["Lead_Time"] + 7,
        ],
        ["Critico", "Preventivo"],
        default="Normal",
    )
    return health, stock, health_daily


@st.cache_resource(show_spinner=False)
def train_risk_model(health: pd.DataFrame) -> tuple[Pipeline, dict[str, float], list[str]]:
    excluded = {"Patient_ID", "Risk_Level", "Overall_Risk_Score"}
    features = [c for c in health.columns if c not in excluded]
    X = health[features]
    y = health["Risk_Level"].astype(str)

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "F1 macro": f1_score(y_test, pred, average="macro"),
        "F1 alto riesgo": f1_score(y_test == "High", pred == "High"),
    }
    return model, metrics, features


@st.cache_resource(show_spinner=False)
def train_stock_model(stock: pd.DataFrame) -> tuple[RandomForestRegressor, dict[str, float], list[str]]:
    model_df = stock.dropna(subset=["Cobertura_Dias"]).copy()
    features = ["Consumo_Diario", "Lead_Time", "Ocupacion_Albergue", "Consumo_7d", "Consumo_14d"]
    X = model_df[features]
    y = model_df["Stock_Actual"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=70, min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": r2_score(y_test, pred),
    }
    return model, metrics, features


def metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def require_plotly() -> bool:
    if px is None or go is None:
        st.error(
            "Falta Plotly. Instala dependencias con: pip install -r requirements.txt"
        )
        return False
    return True


health_df, stock_df, health_daily_df = load_data()
risk_model, risk_metrics, risk_features = train_risk_model(health_df)
stock_model, stock_metrics, stock_features = train_stock_model(stock_df)

st.sidebar.title("ALDIMI Predict")
section = st.sidebar.radio(
    "Vista",
    ["Resumen ejecutivo", "Inventario predictivo", "Priorizacion clinica", "Demo en vivo"],
)
st.sidebar.caption("Prototipo preliminar para exposicion TP - Machine Learning")

st.title("ALDIMI Predict")
st.caption("Dashboard preliminar funcional para alertas de inventario y priorizacion preventiva.")

if section == "Resumen ejecutivo":
    high_risk_count = int((health_df["Risk_Level"].astype(str) == "High").sum())
    critical_items = int((stock_df.groupby("ID_Insumo").tail(1)["Alerta"] == "Critico").sum())
    avg_cover = stock_df.groupby("ID_Insumo").tail(1)["Cobertura_Dias"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros clinicos", f"{len(health_df):,}")
    c2.metric("Pacientes alto riesgo", f"{high_risk_count:,}")
    c3.metric("Insumos criticos hoy", critical_items)
    c4.metric("Cobertura promedio", f"{avg_cover:.1f} dias")

    st.subheader("Estado operativo")
    if require_plotly():
        latest_stock = stock_df.groupby("ID_Insumo").tail(1).sort_values("Cobertura_Dias")
        col_a, col_b = st.columns([1.2, 1])
        with col_a:
            fig = px.bar(
                latest_stock.head(15),
                x="Cobertura_Dias",
                y="ID_Insumo",
                color="Alerta",
                orientation="h",
                color_discrete_map={"Critico": "#c1121f", "Preventivo": "#f48c06", "Normal": "#2a9d8f"},
                labels={"Cobertura_Dias": "Dias de cobertura", "ID_Insumo": "Insumo"},
            )
            fig.update_layout(height=440, margin=dict(l=10, r=10, t=20, b=10), yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, width="stretch")
        with col_b:
            risk_counts = (
                health_df["Risk_Level"].astype(str).map(RISK_LABELS).value_counts().reindex(["Bajo", "Medio", "Alto"])
            )
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                hole=0.5,
                color=risk_counts.index,
                color_discrete_map={"Bajo": "#2a9d8f", "Medio": "#f48c06", "Alto": "#c1121f"},
            )
            fig.update_layout(height=440, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")

    st.subheader("Metricas preliminares de modelos")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("F1 macro riesgo", f"{risk_metrics['F1 macro']:.2%}")
    m2.metric("F1 alto riesgo", f"{risk_metrics['F1 alto riesgo']:.2%}")
    m3.metric("Accuracy riesgo", f"{risk_metrics['Accuracy']:.2%}")
    m4.metric("MAE stock", f"{stock_metrics['MAE']:.1f}")
    m5.metric("RMSE stock", f"{stock_metrics['RMSE']:.1f}")
    m6.metric("R2 stock", f"{stock_metrics['R2']:.2%}")
    st.info(
        "Control de calidad aplicado: el modelo clinico excluye Overall_Risk_Score para evitar data leakage."
    )

elif section == "Inventario predictivo":
    st.subheader("Alertas de abastecimiento")
    selected_item = st.selectbox("Insumo", sorted(stock_df["ID_Insumo"].unique()))
    item_df = stock_df[stock_df["ID_Insumo"] == selected_item].sort_values("Fecha")
    latest = item_df.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stock actual", f"{latest['Stock_Actual']:.0f}")
    c2.metric("Consumo 7d", f"{latest['Consumo_7d']:.1f}")
    c3.metric("Cobertura", f"{latest['Cobertura_Dias']:.1f} dias")
    c4.metric("Alerta", latest["Alerta"])

    if require_plotly():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=item_df["Fecha"], y=item_df["Stock_Actual"], name="Stock actual", line=dict(color="#1d3557")))
        fig.add_trace(go.Scatter(x=item_df["Fecha"], y=item_df["Stock_Proyectado_7d"], name="Proyeccion 7 dias", line=dict(color="#f48c06")))
        fig.add_trace(go.Scatter(x=item_df["Fecha"], y=item_df["Stock_Proyectado_14d"], name="Proyeccion 14 dias", line=dict(color="#c1121f")))
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Unidades")
        st.plotly_chart(fig, width="stretch")

    st.dataframe(
        stock_df.groupby("ID_Insumo")
        .tail(1)
        .sort_values(["Alerta", "Cobertura_Dias"])[
            ["ID_Insumo", "Stock_Actual", "Consumo_7d", "Lead_Time", "Cobertura_Dias", "Stock_Proyectado_7d", "Stock_Proyectado_14d", "Alerta"]
        ],
        width="stretch",
        hide_index=True,
    )

elif section == "Priorizacion clinica":
    st.subheader("Distribucion y factores de riesgo")
    c1, c2 = st.columns([1, 1.15])
    if require_plotly():
        with c1:
            counts = health_df["Risk_Level"].astype(str).map(RISK_LABELS).value_counts().reindex(["Bajo", "Medio", "Alto"])
            fig = px.bar(
                x=counts.index,
                y=counts.values,
                color=counts.index,
                color_discrete_map={"Bajo": "#2a9d8f", "Medio": "#f48c06", "Alto": "#c1121f"},
                labels={"x": "Nivel de prioridad", "y": "Pacientes"},
            )
            fig.update_layout(height=410, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.scatter(
                health_df,
                x="Risk_Lifestyle_Score",
                y="Diet_Risk_Index",
                color=health_df["Risk_Level"].astype(str).map(RISK_LABELS),
                opacity=0.65,
                color_discrete_map={"Bajo": "#2a9d8f", "Medio": "#f48c06", "Alto": "#c1121f"},
                labels={"color": "Prioridad"},
            )
            fig.update_layout(height=410, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")

    st.subheader("Pacientes para seguimiento preventivo")
    st.dataframe(
        health_df[health_df["Risk_Level"].astype(str) == "High"]
        .sort_values(["Risk_Lifestyle_Score", "Diet_Risk_Index"], ascending=False)
        .head(25)[
            ["Patient_ID", "Cancer_Type", "Age", "BMI", "Family_History", "Risk_Lifestyle_Score", "Diet_Risk_Index", "Risk_Level"]
        ],
        width="stretch",
        hide_index=True,
    )

else:
    st.subheader("Simulador de priorizacion")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        age = st.slider("Edad", 1, 90, 14)
        bmi = st.slider("BMI", 14.0, 40.0, 24.0, 0.1)
        family_history = st.toggle("Historial familiar")
        cancer_type = st.selectbox("Tipo de cancer", sorted(health_df["Cancer_Type"].unique()))
    with col_b:
        smoking = st.slider("Tabaquismo / exposicion", 0, 10, 3)
        alcohol = st.slider("Alcohol / exposicion", 0, 10, 2)
        obesity = st.slider("Obesidad", 0, 10, 4)
        physical_activity = st.slider("Actividad fisica", 0, 10, 5)
    with col_c:
        red_meat = st.slider("Dieta carnes rojas", 0, 10, 4)
        salted = st.slider("Procesados / salados", 0, 10, 4)
        fruit = st.slider("Frutas y verduras", 0, 10, 6)
        air = st.slider("Contaminacion", 0, 10, 5)

    sample = health_df.drop(columns=["Patient_ID", "Risk_Level", "Overall_Risk_Score"]).median(numeric_only=True).to_dict()
    sample.update(
        {
            "Cancer_Type": cancer_type,
            "Age": age,
            "BMI": bmi,
            "Family_History": int(family_history),
            "Smoking": smoking,
            "Alcohol_Use": alcohol,
            "Obesity": obesity,
            "Physical_Activity": physical_activity,
            "Physical_Activity_Level": physical_activity,
            "Diet_Red_Meat": red_meat,
            "Diet_Salted_Processed": salted,
            "Fruit_Veg_Intake": fruit,
            "Air_Pollution": air,
            "Risk_Lifestyle_Score": (smoking + alcohol + obesity + air + sample.get("Occupational_Hazards", 5)) / 5,
            "Diet_Risk_Index": (red_meat + salted + (10 - fruit)) / 3,
            "county_CTYNAME": "Demo",
        }
    )
    input_df = pd.DataFrame([sample])[risk_features]
    pred = risk_model.predict(input_df)[0]
    proba = risk_model.predict_proba(input_df)[0]
    proba_df = pd.DataFrame({"Prioridad": [RISK_LABELS.get(c, c) for c in risk_model.classes_], "Probabilidad": proba})

    c1, c2 = st.columns([0.85, 1.15])
    with c1:
        st.metric("Resultado estimado", RISK_LABELS.get(pred, pred))
        st.caption("Herramienta de apoyo preventivo, no diagnostico medico.")
    with c2:
        if require_plotly():
            fig = px.bar(
                proba_df,
                x="Prioridad",
                y="Probabilidad",
                color="Prioridad",
                color_discrete_map={"Bajo": "#2a9d8f", "Medio": "#f48c06", "Alto": "#c1121f"},
            )
            fig.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=20, b=10), yaxis_tickformat=".0%")
            st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Simulador de stock")
    s1, s2, s3, s4 = st.columns(4)
    consumo = s1.number_input("Consumo diario", min_value=0.0, value=18.0, step=1.0)
    lead = s2.number_input("Lead time", min_value=1, value=10, step=1)
    ocupacion = s3.slider("Ocupacion albergue", 0.0, 1.0, 0.7, 0.01)
    consumo14 = s4.number_input("Promedio 14 dias", min_value=0.0, value=20.0, step=1.0)
    stock_pred = stock_model.predict(
        pd.DataFrame(
            [{
                "Consumo_Diario": consumo,
                "Lead_Time": lead,
                "Ocupacion_Albergue": ocupacion,
                "Consumo_7d": consumo,
                "Consumo_14d": consumo14,
            }]
        )[stock_features]
    )[0]
    cobertura = stock_pred / max(consumo, 0.1)
    alerta = "Critico" if cobertura <= lead else "Preventivo" if cobertura <= lead + 7 else "Normal"
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Stock estimado", f"{stock_pred:.0f}")
    sc2.metric("Cobertura estimada", f"{cobertura:.1f} dias")
    sc3.metric("Alerta", alerta)
