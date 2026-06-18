from __future__ import annotations

from io import BytesIO
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

try:
    from xgboost import XGBClassifier, XGBRegressor
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    XGBClassifier = None
    XGBRegressor = None

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
SRC_DIR = BASE_DIR / "src"
HEALTH_FILE = "Dataset_ALDIMI_GravedadPaciente_Enriquecido.csv"
STOCK_FILE = "Dataset_ALDIMI_Logistica_Enriquecido.csv"
RISK_ORDER = ["Low", "Medium", "High"]
RISK_LABELS = {"Low": "Bajo", "Medium": "Medio", "High": "Alto"}
ALERT_COLORS = {"Critico": "#c1121f", "Preventivo": "#f48c06", "Normal": "#2a9d8f"}
RISK_COLORS = {"Bajo": "#2a9d8f", "Medio": "#f48c06", "Alto": "#c1121f"}
REABASTECIMIENTO_ALERT = {0: "Normal", 1: "Preventivo", 2: "Critico"}
RATIO_CRITICO = 2.2
RATIO_PREVENTIVO = 5.6

NAV_GROUPS = {
    "Operacion diaria": ["Resumen ejecutivo", "Inventario predictivo", "Priorizacion clinica"],
    "Analisis de datos": ["Menu estadistico"],
    "Estrategia y gobierno": ["MLOps", "Impacto ODS y etica", "Ecosistema Core AI"],
}

SECTION_DESCRIPTIONS = {
    "Resumen ejecutivo": "Vista unificada con KPIs, semaforos y comparacion de modelos.",
    "Inventario predictivo": "Alertas de stock, proyecciones 7/14 dias y simulador de compras.",
    "Priorizacion clinica": "Clasificacion de riesgo, seguimiento sugerido y simulador de paciente.",
    "Menu estadistico": "Distribuciones, correlaciones, outliers y estadisticas descriptivas.",
    "MLOps": "Arquitectura tecnica, pipeline de datos y despliegue del sistema.",
    "Impacto ODS y etica": "Impacto social estimado, riesgos eticos y mitigaciones.",
    "Ecosistema Core AI": "Integracion con la base comun de datos del proyecto.",
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; }
        .feature-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #eef2f7 100%);
            border-left: 4px solid #1d3557;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.5rem;
        }
        .feature-card h4 { margin: 0 0 0.25rem 0; color: #1d3557; font-size: 0.95rem; }
        .feature-card p { margin: 0; color: #495057; font-size: 0.85rem; }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 0.5rem 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="ALDIMI Core AI",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()


def resolve_data_path(filename: str) -> Path:
    for base in (DATA_DIR, SRC_DIR):
        path = base / filename
        if path.exists():
            return path
    raise FileNotFoundError(f"No se encontro {filename} en data/processed/ ni src/")


def enrich_health_features(health: pd.DataFrame) -> pd.DataFrame:
    health = health.copy()
    if "Habitos_Riesgo" not in health.columns:
        health["Habitos_Riesgo"] = (
            health["Smoking"]
            + health["Alcohol_Use"]
            + health["Obesity"]
            + health["Air_Pollution"]
            + health["Occupational_Hazards"]
        )
    if "Riesgo_Clinico" not in health.columns:
        health["Riesgo_Clinico"] = (
            health["Family_History"] + health["BRCA_Mutation"] + health["H_Pylori_Infection"]
        )
    if "Factor_Protector" not in health.columns:
        health["Factor_Protector"] = (
            health["Fruit_Veg_Intake"] + health["Physical_Activity"] + health["Calcium_Intake"]
        )
    if "Balance_Riesgo" not in health.columns:
        health["Balance_Riesgo"] = health["Habitos_Riesgo"] + health["Riesgo_Clinico"] - health["Factor_Protector"]
    if "Edad_Rango" not in health.columns:
        health["Edad_Rango"] = pd.cut(
            health["Age"],
            bins=[0, 30, 45, 60, 120],
            labels=["Joven", "Adulto", "Mayor", "Adulto_Mayor"],
        )
    health["Risk_Lifestyle_Score"] = health["Habitos_Riesgo"] / 5
    health["Diet_Risk_Index"] = (
        health["Diet_Red_Meat"] + health["Diet_Salted_Processed"] + (10 - health["Fruit_Veg_Intake"])
    ) / 3
    health["Risk_Level"] = pd.Categorical(health["Risk_Level"], categories=RISK_ORDER, ordered=True)
    return health


def alert_from_ratio(ratio: float) -> str:
    if ratio <= RATIO_CRITICO:
        return "Critico"
    if ratio <= RATIO_PREVENTIVO:
        return "Preventivo"
    return "Normal"


def enrich_stock_features(stock: pd.DataFrame) -> pd.DataFrame:
    stock = stock.sort_values(["ID_Insumo", "Fecha"]).copy()
    if "Punto_Reorden" not in stock.columns:
        stock["Punto_Reorden"] = stock["Consumo_Diario"] * stock["Lead_Time"]
    if "Ratio_Stock" not in stock.columns:
        stock["Ratio_Stock"] = stock["Stock_Actual"] / stock["Punto_Reorden"].replace(0, np.nan)
    stock["Consumo_7d"] = stock.groupby("ID_Insumo")["Consumo_Diario"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    stock["Consumo_14d"] = stock.groupby("ID_Insumo")["Consumo_Diario"].transform(
        lambda s: s.rolling(14, min_periods=1).mean()
    )
    stock["Cobertura_Dias"] = stock["Stock_Actual"] / stock["Consumo_7d"].replace(0, np.nan)
    stock["Stock_Proyectado_7d"] = (stock["Stock_Actual"] - (stock["Consumo_7d"] * 7)).clip(lower=0)
    stock["Stock_Proyectado_14d"] = (stock["Stock_Actual"] - (stock["Consumo_14d"] * 14)).clip(lower=0)
    if "Necesita_Reabastecimiento" in stock.columns:
        stock["Alerta"] = stock["Necesita_Reabastecimiento"].map(REABASTECIMIENTO_ALERT)
    else:
        stock["Alerta"] = stock["Ratio_Stock"].apply(alert_from_ratio)
    return stock


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    health = pd.read_csv(resolve_data_path(HEALTH_FILE))
    stock = pd.read_csv(resolve_data_path(STOCK_FILE), parse_dates=["Fecha"])
    health_daily_path = DATA_DIR / "health_daily.csv"
    health_daily = (
        pd.read_csv(health_daily_path, parse_dates=["Fecha"])
        if health_daily_path.exists()
        else pd.DataFrame()
    )

    health = enrich_health_features(health)
    stock = enrich_stock_features(stock)
    return health, stock, health_daily


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def preferred_model(models: dict[str, object]) -> str:
    if "XGBoost" in models:
        return "XGBoost"
    return next(iter(models))


@st.cache_resource(show_spinner=False)
def train_risk_models(
    health: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame, list[str], LabelEncoder, pd.DataFrame, str]:
    excluded = {"Patient_ID", "Risk_Level", "Overall_Risk_Score"}
    features = [c for c in health.columns if c not in excluded]
    X = health[features]
    y = health["Risk_Level"].astype(str)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    models: dict[str, object] = {}
    if XGBClassifier is not None:
        models["XGBoost"] = Pipeline(
            steps=[
                ("prep", make_preprocessor(X)),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=120,
                        max_depth=4,
                        learning_rate=0.08,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        random_state=42,
                    ),
                ),
            ]
        )
    models["Random Forest"] = Pipeline(
        steps=[
            ("prep", make_preprocessor(X)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=140,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    rows = []
    fitted: dict[str, object] = {}
    confusion_rows = []
    high_id = int(np.where(label_encoder.classes_ == "High")[0][0])
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        cm = confusion_matrix(y_test, pred, labels=list(range(len(label_encoder.classes_))))
        high_fn = int(cm[high_id].sum() - cm[high_id, high_id])
        rows.append(
            {
                "Modelo": name,
                "Accuracy": accuracy_score(y_test, pred),
                "F1 macro": f1_score(y_test, pred, average="macro"),
                "F1 alto riesgo": f1_score(y_test == high_id, pred == high_id),
                "Falsos negativos alto": high_fn,
            }
        )
        fitted[name] = model
        for i, actual in enumerate(label_encoder.classes_):
            for j, predicted in enumerate(label_encoder.classes_):
                confusion_rows.append(
                    {
                        "Modelo": name,
                        "Real": RISK_LABELS.get(actual, actual),
                        "Predicho": RISK_LABELS.get(predicted, predicted),
                        "Casos": int(cm[i, j]),
                    }
                )

    metrics = pd.DataFrame(rows).sort_values(
        ["F1 alto riesgo", "F1 macro", "Accuracy"], ascending=False
    )
    confusion_df = pd.DataFrame(confusion_rows)
    selected = metrics.iloc[0]["Modelo"]
    return fitted, metrics, features, label_encoder, confusion_df, selected


@st.cache_resource(show_spinner=False)
def train_stock_models(
    stock: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame, list[str], str]:
    model_df = stock.dropna(subset=["Cobertura_Dias", "Ratio_Stock"]).copy()
    features = [
        "Consumo_Diario",
        "Lead_Time",
        "Ocupacion_Albergue",
        "Consumo_7d",
        "Consumo_14d",
        "Punto_Reorden",
        "Ratio_Stock",
        "Pacientes_Alto_Riesgo",
        "Ocupacion_Total",
    ]
    features = [c for c in features if c in model_df.columns]
    X = model_df[features]
    y = model_df["Stock_Actual"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models: dict[str, object] = {}
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=130,
            max_depth=4,
            learning_rate=0.07,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
    models["Random Forest"] = RandomForestRegressor(
        n_estimators=100, min_samples_leaf=3, random_state=42, n_jobs=-1
    )

    rows = []
    fitted: dict[str, object] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append(
            {
                "Modelo": name,
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                "R2": r2_score(y_test, pred),
            }
        )
        fitted[name] = model

    metrics = pd.DataFrame(rows).sort_values(["MAE", "RMSE"], ascending=True)
    selected = metrics.iloc[0]["Modelo"]
    return fitted, metrics, features, selected


def require_plotly() -> bool:
    if px is None or go is None:
        st.error("Falta Plotly. Instala dependencias con: pip install -r requirements.txt")
        return False
    return True


def format_percent(value: float) -> str:
    return f"{value:.2%}"


def latest_stock(stock: pd.DataFrame) -> pd.DataFrame:
    return stock.groupby("ID_Insumo", as_index=False).tail(1).sort_values("Cobertura_Dias")


def download_csv_button(df: pd.DataFrame, filename: str, label: str = "Descargar CSV") -> None:
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    st.download_button(label, buffer.getvalue(), file_name=filename, mime="text/csv")


def render_feature_cards() -> None:
    cards = [
        ("Priorizacion clinica", "Clasifica pacientes en Bajo, Medio y Alto riesgo con simulador interactivo."),
        ("Inventario predictivo", "Anticipa quiebres con alertas Critico / Preventivo / Normal."),
        ("Menu estadistico", "Explora distribuciones, correlaciones y outliers sin salir del dashboard."),
    ]
    cols = st.columns(3)
    for col, (title, text) in zip(cols, cards):
        col.markdown(
            f'<div class="feature-card"><h4>{title}</h4><p>{text}</p></div>',
            unsafe_allow_html=True,
        )


def render_global_kpis(health: pd.DataFrame, stock: pd.DataFrame) -> None:
    current_stock = latest_stock(stock)
    high_risk = int((health["Risk_Level"].astype(str) == "High").sum())
    critical = int((current_stock["Alerta"] == "Critico").sum())
    preventive = int((current_stock["Alerta"] == "Preventivo").sum())
    avg_cover = current_stock["Cobertura_Dias"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pacientes", f"{len(health):,}")
    c2.metric("Alto riesgo", f"{high_risk:,}")
    c3.metric("Alertas stock", critical + preventive, f"{critical} criticas")
    c4.metric("Cobertura media", f"{avg_cover:.1f} dias")


def render_model_metrics(
    risk_metrics: pd.DataFrame,
    stock_metrics: pd.DataFrame,
    risk_selected: str,
    stock_selected: str,
) -> None:
    st.subheader("Evaluacion comparativa de modelos")
    left, right = st.columns(2)
    with left:
        st.caption(f"Clasificacion de prioridad clinica (modelo en produccion: **{risk_selected}**)")
        display = risk_metrics.copy()
        for col in ["Accuracy", "F1 macro", "F1 alto riesgo"]:
            display[col] = display[col].map(format_percent)
        st.dataframe(display, width="stretch", hide_index=True)
    with right:
        st.caption(f"Regresion de inventario (modelo en produccion: **{stock_selected}**)")
        display = stock_metrics.copy()
        display["MAE"] = display["MAE"].map(lambda x: f"{x:.2f}")
        display["RMSE"] = display["RMSE"].map(lambda x: f"{x:.2f}")
        display["R2"] = display["R2"].map(format_percent)
        st.dataframe(display, width="stretch", hide_index=True)

    if XGBClassifier is None or XGBRegressor is None:
        st.warning(
            "XGBoost no esta instalado. Instala con: pip install xgboost "
            "para activar la comparacion completa y el modelo de produccion."
        )


def render_confusion_heatmap(confusion_df: pd.DataFrame, model_name: str) -> None:
    if not require_plotly():
        return
    subset = confusion_df[confusion_df["Modelo"] == model_name]
    pivot = subset.pivot(index="Real", columns="Predicho", values="Casos").fillna(0)
    order = ["Bajo", "Medio", "Alto"]
    pivot = pivot.reindex(index=order, columns=order).fillna(0)
    fig = px.imshow(
        pivot.values,
        x=pivot.columns,
        y=pivot.index,
        text_auto=True,
        color_continuous_scale="Blues",
        labels={"color": "Casos"},
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch")


def render_executive_view(
    health: pd.DataFrame,
    stock: pd.DataFrame,
    risk_metrics: pd.DataFrame,
    stock_metrics: pd.DataFrame,
    risk_selected: str,
    stock_selected: str,
) -> None:
    render_feature_cards()
    render_global_kpis(health, stock)

    with st.expander("Guia rapida para directivos", expanded=False):
        st.markdown(
            """
            1. **Revise el semaforo operativo** para identificar insumos criticos y distribucion de riesgo.
            2. **Use Inventario predictivo** para planificar reposiciones a 7 o 14 dias.
            3. **Use Priorizacion clinica** para filtrar pacientes de alto riesgo; el modelo es apoyo, no diagnostico.
            """
        )

    if not require_plotly():
        return

    st.subheader("Semaforo operativo")
    col_a, col_b = st.columns([1.2, 1])
    current_stock = latest_stock(stock)
    with col_a:
        data = current_stock.head(15)
        fig = px.bar(
            data,
            x="Cobertura_Dias",
            y="ID_Insumo",
            color="Alerta",
            orientation="h",
            color_discrete_map=ALERT_COLORS,
            labels={"Cobertura_Dias": "Dias de cobertura", "ID_Insumo": "Insumo"},
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10), yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")
    with col_b:
        risk_counts = (
            health["Risk_Level"].astype(str).map(RISK_LABELS).value_counts().reindex(["Bajo", "Medio", "Alto"])
        )
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            hole=0.52,
            color=risk_counts.index,
            color_discrete_map=RISK_COLORS,
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")

    render_model_metrics(risk_metrics, stock_metrics, risk_selected, stock_selected)
    st.info("Control de calidad: se excluye Overall_Risk_Score del modelo clinico para evitar data leakage.")


def render_inventory_view(
    stock: pd.DataFrame,
    stock_models: dict[str, object],
    stock_features: list[str],
    stock_selected: str,
) -> None:
    st.subheader("Inventario predictivo")
    c1, c2, c3 = st.columns([1, 1, 1])
    selected_item = c1.selectbox("Insumo", sorted(stock["ID_Insumo"].unique()))
    horizon = c2.radio("Horizonte", ["7 dias", "14 dias"], horizontal=True)
    alert_filter = c3.multiselect(
        "Alertas", ["Critico", "Preventivo", "Normal"], default=["Critico", "Preventivo", "Normal"]
    )

    item_df = stock[stock["ID_Insumo"] == selected_item].sort_values("Fecha")
    latest = item_df.iloc[-1]
    projected_col = "Stock_Proyectado_7d" if horizon == "7 dias" else "Stock_Proyectado_14d"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stock actual", f"{latest['Stock_Actual']:.0f}")
    m2.metric("Ratio stock", f"{latest['Ratio_Stock']:.2f}")
    m3.metric("Punto reorden", f"{latest['Punto_Reorden']:.0f}")
    m4.metric("Alerta", latest["Alerta"])

    if require_plotly():
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=item_df["Fecha"],
                y=item_df["Stock_Actual"],
                name="Stock actual",
                line=dict(color="#1d3557"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=item_df["Fecha"],
                y=item_df[projected_col],
                name=f"Proyeccion {horizon}",
                line=dict(color="#c1121f"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=item_df["Fecha"],
                y=item_df["Punto_Reorden"],
                name="Punto reorden",
                line=dict(color="#f48c06", dash="dash"),
            )
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Unidades")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Lista priorizada de reposicion")
    table = latest_stock(stock)
    table = table[table["Alerta"].isin(alert_filter)]
    st.dataframe(
        table[
            [
                "ID_Insumo",
                "Stock_Actual",
                "Consumo_7d",
                "Lead_Time",
                "Punto_Reorden",
                "Ratio_Stock",
                "Cobertura_Dias",
                "Stock_Proyectado_7d",
                "Stock_Proyectado_14d",
                "Necesita_Reabastecimiento",
                "Alerta",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    download_csv_button(table, "reposicion_priorizada.csv")

    with st.expander("Simulador de compras y cobertura", expanded=True):
        s1, s2, s3, s4 = st.columns(4)
        consumo = s1.number_input("Consumo diario", min_value=0.0, value=float(latest["Consumo_Diario"]), step=1.0)
        lead = s2.number_input("Lead time", min_value=1, value=int(latest["Lead_Time"]), step=1)
        ocupacion = s3.slider(
            "Ocupacion del albergue", 0.0, 1.0, float(latest.get("Ocupacion_Albergue", 0.7)), 0.01
        )
        consumo14 = s4.number_input("Promedio 14 dias", min_value=0.0, value=float(latest["Consumo_14d"]), step=1.0)
        pacientes_alto = st.number_input(
            "Pacientes alto riesgo",
            min_value=0,
            value=int(latest.get("Pacientes_Alto_Riesgo", 0)),
            step=1,
        )
        ocupacion_total = st.number_input(
            "Ocupacion total albergue",
            min_value=0,
            value=int(latest.get("Ocupacion_Total", 70)),
            step=1,
        )
        model_names = list(stock_models.keys())
        model_name = st.selectbox(
            "Modelo",
            model_names,
            index=model_names.index(stock_selected) if stock_selected in model_names else 0,
        )
        punto_reorden = consumo * lead
        ratio_input = latest["Stock_Actual"] / max(punto_reorden, 0.1)
        stock_row = {
            "Consumo_Diario": consumo,
            "Lead_Time": lead,
            "Ocupacion_Albergue": ocupacion,
            "Consumo_7d": consumo,
            "Consumo_14d": consumo14,
            "Punto_Reorden": punto_reorden,
            "Ratio_Stock": ratio_input,
            "Pacientes_Alto_Riesgo": pacientes_alto,
            "Ocupacion_Total": ocupacion_total,
        }
        stock_pred = stock_models[model_name].predict(pd.DataFrame([stock_row])[stock_features])[0]
        ratio_pred = stock_pred / max(punto_reorden, 0.1)
        alerta = alert_from_ratio(ratio_pred)
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Stock estimado", f"{stock_pred:.0f}")
        r2.metric("Ratio estimado", f"{ratio_pred:.2f}")
        r3.metric("Punto reorden", f"{punto_reorden:.0f}")
        r4.metric("Alerta", alerta)
        if alerta == "Critico":
            st.error("Accion recomendada: iniciar reposicion inmediata antes del lead time.")
        elif alerta == "Preventivo":
            st.warning("Accion recomendada: planificar pedido en los proximos dias.")
        else:
            st.success("Nivel de cobertura dentro de parametros operativos.")


def render_clinical_view(
    health: pd.DataFrame,
    risk_models: dict[str, object],
    risk_features: list[str],
    label_encoder: LabelEncoder,
    confusion_df: pd.DataFrame,
    risk_selected: str,
) -> None:
    st.subheader("Priorizacion preventiva")
    c1, c2, c3 = st.columns(3)
    level_filter = c1.multiselect("Niveles", ["Bajo", "Medio", "Alto"], default=["Bajo", "Medio", "Alto"])
    cancer_filter = c2.multiselect(
        "Tipo de cancer",
        sorted(health["Cancer_Type"].unique()),
        default=sorted(health["Cancer_Type"].unique())[:3],
    )
    family_only = c3.toggle("Solo con historial familiar", value=False)

    selected_raw_levels = [k for k, v in RISK_LABELS.items() if v in level_filter]
    filtered = health[
        health["Risk_Level"].astype(str).isin(selected_raw_levels) & health["Cancer_Type"].isin(cancer_filter)
    ]
    if family_only:
        filtered = filtered[filtered["Family_History"] == 1]

    if require_plotly():
        left, right = st.columns([1, 1.15])
        with left:
            counts = (
                filtered["Risk_Level"].astype(str).map(RISK_LABELS).value_counts().reindex(level_filter).fillna(0)
            )
            fig = px.bar(
                x=counts.index,
                y=counts.values,
                color=counts.index,
                color_discrete_map=RISK_COLORS,
                labels={"x": "Prioridad", "y": "Pacientes"},
            )
            fig.update_layout(height=390, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
        with right:
            fig = px.scatter(
                filtered,
                x="Habitos_Riesgo",
                y="Balance_Riesgo",
                color=filtered["Risk_Level"].astype(str).map(RISK_LABELS),
                opacity=0.65,
                color_discrete_map=RISK_COLORS,
                labels={"color": "Prioridad", "Habitos_Riesgo": "Habitos de riesgo", "Balance_Riesgo": "Balance de riesgo"},
                hover_data=["Patient_ID", "Age", "BMI", "Factor_Protector"],
            )
            fig.update_layout(height=390, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")

    st.subheader("Pacientes sugeridos para seguimiento")
    follow_up = filtered.sort_values(["Balance_Riesgo", "Habitos_Riesgo"], ascending=False).head(30)
    st.dataframe(
        follow_up[
            [
                "Patient_ID",
                "Cancer_Type",
                "Age",
                "Edad_Rango",
                "BMI",
                "Family_History",
                "Habitos_Riesgo",
                "Riesgo_Clinico",
                "Factor_Protector",
                "Balance_Riesgo",
                "Risk_Level",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    download_csv_button(follow_up, "seguimiento_clinico.csv")

    with st.expander("Matriz de confusion del modelo (validacion)"):
        model_names = list(risk_models.keys())
        model_name = st.selectbox(
            "Modelo para matriz",
            model_names,
            index=model_names.index(risk_selected) if risk_selected in model_names else 0,
            key="cm_model",
        )
        render_confusion_heatmap(confusion_df, model_name)

    with st.expander("Simulador de prioridad de paciente", expanded=True):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            age = st.slider("Edad", 1, 90, 14)
            bmi = st.slider("BMI", 14.0, 40.0, 24.0, 0.1)
            family_history = st.toggle("Historial familiar")
            cancer_type = st.selectbox("Tipo de cancer", sorted(health["Cancer_Type"].unique()))
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

        model_names = list(risk_models.keys())
        model_name = st.selectbox(
            "Modelo de clasificacion",
            model_names,
            index=model_names.index(risk_selected) if risk_selected in model_names else 0,
        )
        sample = health.drop(columns=["Patient_ID", "Risk_Level", "Overall_Risk_Score"]).median(numeric_only=True).to_dict()
        habitos = smoking + alcohol + obesity + air + int(sample.get("Occupational_Hazards", 5))
        riesgo_clinico = int(family_history) + int(sample.get("BRCA_Mutation", 0)) + int(sample.get("H_Pylori_Infection", 0))
        factor_protector = fruit + physical_activity + int(sample.get("Calcium_Intake", 5))
        balance = habitos + riesgo_clinico - factor_protector
        edad_rango = pd.cut(
            [age], bins=[0, 30, 45, 60, 120], labels=["Joven", "Adulto", "Mayor", "Adulto_Mayor"]
        )[0]
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
                "Habitos_Riesgo": habitos,
                "Riesgo_Clinico": riesgo_clinico,
                "Factor_Protector": factor_protector,
                "Balance_Riesgo": balance,
                "Edad_Rango": str(edad_rango),
                "Risk_Lifestyle_Score": habitos / 5,
                "Diet_Risk_Index": (red_meat + salted + (10 - fruit)) / 3,
                "county_CTYNAME": "Demo",
            }
        )
        input_df = pd.DataFrame([sample])[risk_features]
        pred_id = int(risk_models[model_name].predict(input_df)[0])
        proba = risk_models[model_name].predict_proba(input_df)[0]
        classes = label_encoder.inverse_transform(np.arange(len(proba)))
        proba_df = pd.DataFrame({"Prioridad": [RISK_LABELS.get(c, c) for c in classes], "Probabilidad": proba})
        pred_label = RISK_LABELS.get(label_encoder.inverse_transform([pred_id])[0], "N/D")

        r1, r2 = st.columns([0.85, 1.15])
        with r1:
            st.metric("Resultado estimado", pred_label)
            st.caption("Herramienta de apoyo preventivo; no reemplaza evaluacion medica.")
        with r2:
            if require_plotly():
                fig = px.bar(
                    proba_df,
                    x="Prioridad",
                    y="Probabilidad",
                    color="Prioridad",
                    color_discrete_map=RISK_COLORS,
                )
                fig.update_layout(
                    height=300, showlegend=False, margin=dict(l=10, r=10, t=20, b=10), yaxis_tickformat=".0%"
                )
                st.plotly_chart(fig, width="stretch")


def compute_zscore_outliers(series: pd.Series, threshold: float = 3.0) -> pd.DataFrame:
    clean = series.dropna()
    std = clean.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.DataFrame(columns=["Valor", "Z-score"])
    z = (clean - clean.mean()) / std
    flagged = clean[np.abs(z) > threshold]
    return pd.DataFrame({"Valor": flagged.values, "Z-score": z[np.abs(z) > threshold].values}).head(20)


def render_statistics_view(health: pd.DataFrame, stock: pd.DataFrame) -> None:
    st.subheader("Menu estadistico")
    st.caption(
        "Exploracion interactiva alineada con el EDA del informe: "
        "distribuciones, correlaciones, outliers y resumen."
    )

    dataset = st.radio("Dataset", ["Clinico", "Inventario"], horizontal=True)
    stat_view = st.selectbox(
        "Tipo de analisis",
        [
            "Distribuciones",
            "Correlaciones",
            "Dispersion",
            "Outliers (Z-score)",
            "Resumen descriptivo",
            "Analisis por categoria",
        ],
    )

    if not require_plotly():
        return

    if dataset == "Clinico":
        if stat_view == "Distribuciones":
            variable = st.selectbox(
                "Variable",
                [
                    "Age",
                    "BMI",
                    "Habitos_Riesgo",
                    "Riesgo_Clinico",
                    "Factor_Protector",
                    "Balance_Riesgo",
                    "Risk_Lifestyle_Score",
                    "Diet_Risk_Index",
                ],
            )
            fig = px.histogram(
                health,
                x=variable,
                color=health["Risk_Level"].astype(str).map(RISK_LABELS),
                color_discrete_map=RISK_COLORS,
                marginal="box",
            )
            fig.update_layout(height=480, margin=dict(l=10, r=10, t=20, b=10), bargap=0.05)
            st.plotly_chart(fig, width="stretch")
        elif stat_view == "Correlaciones":
            cols = [
                "Age",
                "BMI",
                "Smoking",
                "Alcohol_Use",
                "Obesity",
                "Family_History",
                "Air_Pollution",
                "Habitos_Riesgo",
                "Riesgo_Clinico",
                "Factor_Protector",
                "Balance_Riesgo",
                "Risk_Lifestyle_Score",
                "Diet_Risk_Index",
            ]
            fig = px.imshow(health[cols].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
            fig.update_layout(height=620, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
        elif stat_view == "Dispersion":
            x_var = st.selectbox("Eje X", ["Age", "BMI", "Habitos_Riesgo", "Balance_Riesgo", "Risk_Lifestyle_Score"])
            y_var = st.selectbox("Eje Y", ["Factor_Protector", "Riesgo_Clinico", "Diet_Risk_Index", "BMI"])
            fig = px.scatter(
                health,
                x=x_var,
                y=y_var,
                color=health["Risk_Level"].astype(str).map(RISK_LABELS),
                color_discrete_map=RISK_COLORS,
                opacity=0.6,
            )
            fig.update_layout(height=480, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
        elif stat_view == "Outliers (Z-score)":
            variable = st.selectbox(
                "Variable numerica",
                ["Age", "BMI", "Balance_Riesgo", "Habitos_Riesgo", "Factor_Protector", "Diet_Risk_Index"],
            )
            threshold = st.slider("Umbral |Z|", 2.0, 4.0, 3.0, 0.1)
            outliers = compute_zscore_outliers(health[variable], threshold)
            st.metric("Outliers detectados", len(outliers))
            st.dataframe(outliers, width="stretch", hide_index=True)
            fig = px.box(
                health,
                y=variable,
                color=health["Risk_Level"].astype(str).map(RISK_LABELS),
                color_discrete_map=RISK_COLORS,
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
        elif stat_view == "Resumen descriptivo":
            st.dataframe(health.describe(include="all").transpose(), width="stretch")
            download_csv_button(health.describe(include="all").transpose().reset_index(), "resumen_clinico.csv")
        else:
            view = st.radio("Categoria", ["Historial familiar vs riesgo", "Top tipos de cancer"], horizontal=True)
            if view == "Historial familiar vs riesgo":
                grouped = (
                    health.groupby(["Family_History", health["Risk_Level"].astype(str).map(RISK_LABELS)])
                    .size()
                    .reset_index(name="Pacientes")
                )
                grouped.columns = ["Historial familiar", "Prioridad", "Pacientes"]
                fig = px.bar(
                    grouped,
                    x="Historial familiar",
                    y="Pacientes",
                    color="Prioridad",
                    barmode="group",
                    color_discrete_map=RISK_COLORS,
                )
            else:
                top = health["Cancer_Type"].value_counts().head(8).reset_index()
                top.columns = ["Cancer_Type", "Pacientes"]
                fig = px.bar(top, x="Cancer_Type", y="Pacientes", color_discrete_sequence=["#1d3557"])
            fig.update_layout(height=460, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
    else:
        if stat_view == "Distribuciones":
            item = st.selectbox("Insumo", sorted(stock["ID_Insumo"].unique()))
            variable = st.selectbox(
                "Variable",
                ["Consumo_Diario", "Stock_Actual", "Ratio_Stock", "Punto_Reorden", "Cobertura_Dias"],
            )
            fig = px.histogram(
                stock[stock["ID_Insumo"] == item],
                x=variable,
                nbins=35,
                color_discrete_sequence=["#1d3557"],
                marginal="box",
            )
            fig.update_layout(height=480, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
        elif stat_view == "Correlaciones":
            cols = [
                "Consumo_Diario",
                "Stock_Actual",
                "Lead_Time",
                "Consumo_7d",
                "Consumo_14d",
                "Punto_Reorden",
                "Ratio_Stock",
                "Cobertura_Dias",
                "Ocupacion_Albergue",
                "Pacientes_Alto_Riesgo",
                "Ocupacion_Total",
            ]
            fig = px.imshow(stock[cols].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
            fig.update_layout(height=620, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
        elif stat_view == "Dispersion":
            fig = px.scatter(
                stock,
                x="Consumo_Diario",
                y="Stock_Actual",
                color="Alerta",
                color_discrete_map=ALERT_COLORS,
                hover_data=["Ratio_Stock", "Punto_Reorden"],
                opacity=0.5,
            )
            fig.update_layout(height=480, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
        elif stat_view == "Outliers (Z-score)":
            variable = st.selectbox(
                "Variable", ["Stock_Actual", "Consumo_Diario", "Ratio_Stock", "Cobertura_Dias", "Lead_Time"]
            )
            threshold = st.slider("Umbral |Z|", 2.0, 4.0, 3.0, 0.1, key="stock_zscore")
            outliers = compute_zscore_outliers(stock[variable], threshold)
            st.metric("Outliers detectados", len(outliers))
            st.dataframe(outliers, width="stretch", hide_index=True)
            fig = px.box(stock.dropna(subset=[variable]), x="ID_Insumo", y=variable)
            fig.update_layout(height=460, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
        elif stat_view == "Resumen descriptivo":
            st.dataframe(stock.describe(include="all").transpose(), width="stretch")
            download_csv_button(stock.describe(include="all").transpose().reset_index(), "resumen_inventario.csv")
        else:
            current = latest_stock(stock)
            fig = px.bar(
                current, x="ID_Insumo", y="Ratio_Stock", color="Alerta", color_discrete_map=ALERT_COLORS
            )
            fig.update_layout(height=460, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")


def render_mlops_view(
    risk_metrics: pd.DataFrame,
    stock_metrics: pd.DataFrame,
    risk_selected: str,
    stock_selected: str,
) -> None:
    st.subheader("Implementacion y Despliegue (MLOps)")
    st.markdown(
        """
        Pipeline end-to-end que conecta la **base comun de datos** del curso con modelos supervisados
        y el dashboard **ALDIMI Core AI** para convertir analitica en alertas operativas.
        """
    )

    st.markdown("#### Flujo del pipeline")
    st.code(
        """
[Kaggle / API] --> data/raw/
      |
      v
[Notebooks: merge, EDA, enriquecimiento] --> data/processed/
      |
      +--> Dataset_ALDIMI_GravedadPaciente_Enriquecido.csv
      +--> Dataset_ALDIMI_Logistica_Enriquecido.csv
      |
      v
[Modelos: Random Forest + XGBoost] --> metricas (F1, MAE, RMSE, R2)
      |
      v
[Streamlit Dashboard] --> KPIs, simuladores, menu estadistico, alertas
        """,
        language="text",
    )

    architecture = pd.DataFrame(
        [
            ["Capa de datos", "Ingestion y versionado de CSV enriquecidos", HEALTH_FILE + ", " + STOCK_FILE],
            ["Capa analitica", "Feature engineering (notebook 06)", "Habitos_Riesgo, Ratio_Stock, Necesita_Reabastecimiento"],
            ["Capa de modelos", "Clasificacion y regresion supervisada", "scikit-learn, xgboost"],
            ["Capa de servicio", "Entrenamiento en cache y prediccion en vivo", "streamlit_app.py (@st.cache_resource)"],
            ["Capa de presentacion", "Visualizacion interactiva y simuladores", "Streamlit + Plotly"],
            ["Capa de gobierno", "Metricas, trazabilidad y control de leakage", "tablas de evaluacion en dashboard"],
        ],
        columns=["Capa", "Funcion", "Artefacto"],
    )
    st.dataframe(architecture, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Stack tecnologico")
        st.markdown(
            """
            | Componente | Tecnologia |
            |---|---|
            | Lenguaje | Python 3.10+ |
            | Datos | Pandas, NumPy |
            | ML | scikit-learn, XGBoost |
            | UI | Streamlit |
            | Graficos | Plotly |
            | Control de versiones | Git / GitHub |
            """
        )
    with c2:
        st.markdown("#### Despliegue recomendado")
        st.markdown(
            """
            1. Clonar repositorio y crear entorno virtual.
            2. `pip install -r requirements.txt`
            3. `streamlit run streamlit_app.py`
            4. Para demo academica: **Streamlit Community Cloud**.
            5. Para produccion: contenedor Docker + datos anonimizados + autenticacion.
            """
        )

    st.markdown("#### Modelos en produccion")
    st.markdown(
        f"- **Clasificacion clinica:** {risk_selected} (seleccionado por mejor F1 en clase Alto Riesgo)\n"
        f"- **Regresion de inventario:** {stock_selected} (seleccionado por menor MAE)"
    )

    st.markdown("#### Metricas de monitoreo en produccion")
    render_model_metrics(risk_metrics, stock_metrics, risk_selected, stock_selected)
    st.info(
        "Antes de uso operativo con datos reales de ALDIMI: anonimizar identificadores, "
        "registrar version del modelo y establecer revision humana obligatoria en alertas clinicas."
    )


def render_impact_ethics_view(health: pd.DataFrame, stock: pd.DataFrame) -> None:
    st.subheader("Analisis de Impacto ODS y Etica de Datos")

    current_stock = latest_stock(stock)
    critical_rate = (current_stock["Alerta"] == "Critico").mean()
    high_risk_rate = (health["Risk_Level"].astype(str) == "High").mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ODS 3", "Salud y bienestar", "Priorizacion temprana")
    c2.metric("ODS 10", "Reduccion desigualdades", "Variables socioeconomicas")
    c3.metric("Mejora logistica*", "~28.5%", "Rupturas de stock (literatura)")
    c4.metric("Precision predictiva*", "~30.5%", "vs metodos tradicionales")

    st.caption("*Estimaciones basadas en Chirinos & Vereau (2025) para inventarios farmaceuticos predictivos.")

    st.markdown("#### Impacto operativo estimado en ALDIMI 2.0")
    impact = pd.DataFrame(
        [
            [
                "ODS 3 - Salud",
                "Deteccion temprana de pacientes prioritarios",
                f"{high_risk_rate:.1%} del dataset en alto riesgo identificable",
            ],
            [
                "ODS 10 - Equidad",
                "Perfil 360 con variables de condado",
                "Reduce sesgo por falta de contexto social",
            ],
            [
                "Logistica",
                "Alertas preventivas de inventario",
                f"{critical_rate:.1%} de insumos en alerta critica hoy",
            ],
            [
                "Eficiencia",
                "Automatizacion de analisis manual",
                "Libera tiempo del equipo para acompanamiento directo",
            ],
            [
                "Seguridad alimentaria",
                "Control de insumos perecederos",
                "Proyecciones 7/14 dias para evitar desperdicio",
            ],
        ],
        columns=["Dimension ODS", "Beneficio", "Indicador / evidencia"],
    )
    st.dataframe(impact, width="stretch", hide_index=True)

    st.markdown("#### Matriz de riesgos eticos y mitigaciones")
    ethics = pd.DataFrame(
        [
            [
                "Sesgo por datos publicos no pediatricos",
                "Alto",
                "Validar con datos reales anonimizados; no usar como diagnostico.",
            ],
            [
                "Falso negativo en alto riesgo",
                "Alto",
                "Priorizar F1 alto riesgo; revision humana obligatoria.",
            ],
            [
                "Data leakage (Overall_Risk_Score)",
                "Alto",
                "Variable excluida del modelo; auditoria continua.",
            ],
            [
                "Privacidad de pacientes",
                "Alto",
                "Anonimizacion, minimo dato necesario, control de acceso.",
            ],
            [
                "Sobreconfianza en probabilidades",
                "Medio",
                "Mostrar incertidumbre y disclaimers en simuladores.",
            ],
            [
                "Desbalance de clases",
                "Medio",
                "SMOTE en entrenamiento + metricas macro y por clase.",
            ],
        ],
        columns=["Riesgo", "Criticidad", "Mitigacion"],
    )
    st.dataframe(ethics, width="stretch", hide_index=True)

    st.warning(
        "El sistema es una herramienta de apoyo a la decision. Toda alerta clinica debe ser validada "
        "por el equipo medico y social de ALDIMI antes de cualquier accion."
    )


def render_ecosystem_view() -> None:
    st.subheader('Confluencia con el Ecosistema "ALDIMI Core AI"')

    st.markdown(
        """
        ALDIMI Core AI integra el trabajo del equipo de Machine Learning con la **base comun de datos**
        del curso de Inteligencia Artificial, permitiendo que cada modulo aporte datos procesados
        que el dashboard consume de forma unificada.
        """
    )

    flow = pd.DataFrame(
        [
            ["01_Adquisicion_Datos.ipynb", "Ingestion desde Kaggle (salud + inventario)", "data/raw/"],
            ["06_Enriquecimiento_Preparacion de Datos.ipynb", "Features clinicas y logisticas", HEALTH_FILE + ", " + STOCK_FILE],
            ["04/05 Modelado", "Entrenamiento RF vs XGBoost", "Metricas y comparacion"],
            ["streamlit_app.py", "Capa de negocio: KPIs, alertas, simuladores", "Dashboard ALDIMI Core AI"],
        ],
        columns=["Modulo", "Responsabilidad", "Salida"],
    )
    st.dataframe(flow, width="stretch", hide_index=True)

    st.markdown("#### Integracion entre equipos")
    integration = pd.DataFrame(
        [
            ["Base comun IA", "Esquema unificado de pacientes, insumos y ocupacion", "Diccionario de datos compartido"],
            ["Motor ML (equipo modelado)", "Modelos entrenados y metricas de evaluacion", "Predicciones y alertas"],
            ["Dashboard (Fabian)", "Traduccion a lenguaje de negocio", "Streamlit interactivo"],
            ["Usuarios ALDIMI", "Direccion, logistica, asistencia social", "Decisiones preventivas"],
            ["Retroalimentacion", "Nuevos registros operativos", "Reentrenamiento periodico del modelo"],
        ],
        columns=["Actor / componente", "Aporte", "Resultado"],
    )
    st.dataframe(integration, width="stretch", hide_index=True)

    st.markdown("#### Lecciones aprendidas en la integracion")
    st.markdown(
        """
        - **Estandarizar nombres de columnas** desde el merge evita errores al cargar CSV en el dashboard.
        - **Separar datos procesados de notebooks** (`data/processed/`) desacopla experimentacion de la interfaz.
        - **Detectar data leakage temprano** (Overall_Risk_Score) evita metricas enganosas en produccion.
        - **Documentar supuestos** (datos publicos vs operativos) mantiene expectativas realistas con ALDIMI.
        - **Versionar modelos y datasets** es prerequisito antes de escalar de 50 a 100 familias.
        """
    )
    st.info("Repositorio del proyecto: https://github.com/SebasUPC/Machine-learning")


def render_sidebar() -> str:
    st.sidebar.title("ALDIMI Core AI")
    st.sidebar.caption("Ecosistema de gestion inteligente")

    group = st.sidebar.selectbox("Area", list(NAV_GROUPS.keys()))
    section = st.sidebar.radio("Seccion", NAV_GROUPS[group])
    st.sidebar.divider()
    st.sidebar.markdown(f"**{section}**")
    st.sidebar.caption(SECTION_DESCRIPTIONS[section])
    st.sidebar.divider()
    st.sidebar.markdown("**Caracteristicas clave**")
    st.sidebar.markdown("- Alertas de stock en tiempo real")
    st.sidebar.markdown("- Priorizacion clinica con simulador")
    st.sidebar.markdown("- Menu estadistico interactivo")
    st.sidebar.markdown("- Gobernanza MLOps y etica")
    return section


health_df, stock_df, _health_daily_df = load_data()
risk_models, risk_metrics_df, risk_features, label_encoder, confusion_df, risk_selected = train_risk_models(
    health_df
)
stock_models, stock_metrics_df, stock_features, stock_selected = train_stock_models(stock_df)

section = render_sidebar()

st.title("ALDIMI Core AI")
st.caption(SECTION_DESCRIPTIONS[section])

if section == "Resumen ejecutivo":
    render_executive_view(
        health_df, stock_df, risk_metrics_df, stock_metrics_df, risk_selected, stock_selected
    )
elif section == "Inventario predictivo":
    render_inventory_view(stock_df, stock_models, stock_features, stock_selected)
elif section == "Priorizacion clinica":
    render_clinical_view(
        health_df, risk_models, risk_features, label_encoder, confusion_df, risk_selected
    )
elif section == "Menu estadistico":
    render_statistics_view(health_df, stock_df)
elif section == "MLOps":
    render_mlops_view(risk_metrics_df, stock_metrics_df, risk_selected, stock_selected)
elif section == "Impacto ODS y etica":
    render_impact_ethics_view(health_df, stock_df)
else:
    render_ecosystem_view()
