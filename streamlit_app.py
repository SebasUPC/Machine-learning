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
DB_PATH = DATA_DIR / "aldimi.db"

# Inicializar Base de Datos SQLite
try:
    db_path_str = str(DB_PATH)
    
    def get_csv_path(filename):
        for base in (DATA_DIR, SRC_DIR):
            path = base / filename
            if path.exists():
                return str(path)
        return ""
        
    health_csv_str = get_csv_path(HEALTH_FILE)
    stock_csv_str = get_csv_path(STOCK_FILE)
    
    import db_infrastructure as db
    db.init_db(db_path_str, health_csv_str, stock_csv_str)
except Exception as e:
    import streamlit as st
    st.error(f"Error al inicializar la base de datos SQLite: {e}")
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


def predict_horizon_stock(item_df: pd.DataFrame, horizon_days: int, model, features: list[str]) -> np.ndarray:
    """Predice el stock futuro a un horizonte específico usando el modelo de ML y variables proyectadas."""
    projected_features = []
    for _, row in item_df.iterrows():
        # Calcular consumo estimado acumulado para re-evaluar el ratio proyectado
        consumo_ref = row["Consumo_7d"] if horizon_days == 7 else row["Consumo_14d"]
        proj_stock_basic = max(0, row["Stock_Actual"] - (consumo_ref * horizon_days))
        ratio_stock_proj = proj_stock_basic / max(row["Punto_Reorden"], 0.1)
        
        feature_row = {
            "Consumo_Diario": row["Consumo_Diario"],
            "Lead_Time": row["Lead_Time"],
            "Ocupacion_Albergue": row.get("Ocupacion_Albergue", 0.7),
            "Consumo_7d": row["Consumo_7d"],
            "Consumo_14d": row["Consumo_14d"],
            "Punto_Reorden": row["Punto_Reorden"],
            "Ratio_Stock": ratio_stock_proj,
            "Pacientes_Alto_Riesgo": row.get("Pacientes_Alto_Riesgo", 0),
            "Ocupacion_Total": row.get("Ocupacion_Total", 70)
        }
        projected_features.append(feature_row)
        
    df_feat = pd.DataFrame(projected_features)[features]
    preds = model.predict(df_feat)
    return np.clip(preds, 0, None)


def check_active_alerts(stock: pd.DataFrame, model, features: list[str]) -> list[dict]:
    """Genera alertas activas usando predicciones del modelo de ML para el horizonte de 7 y 14 días."""
    latest = latest_stock(stock)
    alerts = []
    for _, row in latest.iterrows():
        # Predicción a 7 días
        proj_stock_7 = max(0, row["Stock_Actual"] - (row["Consumo_7d"] * 7))
        ratio_7 = proj_stock_7 / max(row["Punto_Reorden"], 0.1)
        feat_7 = {
            "Consumo_Diario": row["Consumo_Diario"],
            "Lead_Time": row["Lead_Time"],
            "Ocupacion_Albergue": row.get("Ocupacion_Albergue", 0.7),
            "Consumo_7d": row["Consumo_7d"],
            "Consumo_14d": row["Consumo_14d"],
            "Punto_Reorden": row["Punto_Reorden"],
            "Ratio_Stock": ratio_7,
            "Pacientes_Alto_Riesgo": row.get("Pacientes_Alto_Riesgo", 0),
            "Ocupacion_Total": row.get("Ocupacion_Total", 70)
        }
        pred_stock_7 = max(0, float(model.predict(pd.DataFrame([feat_7])[features])[0]))
        
        # Predicción a 14 días
        proj_stock_14 = max(0, row["Stock_Actual"] - (row["Consumo_14d"] * 14))
        ratio_14 = proj_stock_14 / max(row["Punto_Reorden"], 0.1)
        feat_14 = {
            "Consumo_Diario": row["Consumo_Diario"],
            "Lead_Time": row["Lead_Time"],
            "Ocupacion_Albergue": row.get("Ocupacion_Albergue", 0.7),
            "Consumo_7d": row["Consumo_7d"],
            "Consumo_14d": row["Consumo_14d"],
            "Punto_Reorden": row["Punto_Reorden"],
            "Ratio_Stock": ratio_14,
            "Pacientes_Alto_Riesgo": row.get("Pacientes_Alto_Riesgo", 0),
            "Ocupacion_Total": row.get("Ocupacion_Total", 70)
        }
        pred_stock_14 = max(0, float(model.predict(pd.DataFrame([feat_14])[features])[0]))
        
        # Determinar severidad
        if pred_stock_7 <= 0:
            alerts.append({
                "insumo": row["ID_Insumo"],
                "horizonte": "7 días",
                "severidad": "Crítico",
                "mensaje": f"⚠️ **CRÍTICO ({row['ID_Insumo'].upper()}):** El modelo predice **desabastecimiento total** en 7 días (Stock predicho: {pred_stock_7:.0f} u., Lead Time: {row['Lead_Time']} días).",
                "stock_pred": pred_stock_7
            })
        elif pred_stock_7 <= row["Punto_Reorden"]:
            alerts.append({
                "insumo": row["ID_Insumo"],
                "horizonte": "7 días",
                "severidad": "Preventivo",
                "mensaje": f"⚠️ **PREVENTIVO ({row['ID_Insumo'].upper()}):** El modelo predice caída bajo el **punto de reorden** en 7 días (Stock predicho: {pred_stock_7:.0f} u. vs Reorden: {row['Punto_Reorden']:.0f} u.).",
                "stock_pred": pred_stock_7
            })
            
        if pred_stock_14 <= 0:
            alerts.append({
                "insumo": row["ID_Insumo"],
                "horizonte": "14 días",
                "severidad": "Crítico",
                "mensaje": f"⚠️ **CRÍTICO ({row['ID_Insumo'].upper()}):** El modelo predice **desabastecimiento total** en 14 días (Stock predicho: {pred_stock_14:.0f} u.).",
                "stock_pred": pred_stock_14
            })
        elif pred_stock_14 <= row["Punto_Reorden"]:
            alerts.append({
                "insumo": row["ID_Insumo"],
                "horizonte": "14 días",
                "severidad": "Preventivo",
                "mensaje": f"⚠️ **PREVENTIVO ({row['ID_Insumo'].upper()}):** El modelo predice caída bajo el **punto de reorden** en 14 días (Stock predicho: {pred_stock_14:.0f} u. vs Reorden: {row['Punto_Reorden']:.0f} u.).",
                "stock_pred": pred_stock_14
            })
    return alerts


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import db_infrastructure as db
    db_path_str = str(DB_PATH)
    
    health = db.fetch_all_pacientes(db_path_str)
    stock = db.fetch_all_inventario(db_path_str)
    
    if health.empty:
        health = pd.read_csv(resolve_data_path(HEALTH_FILE))
    if stock.empty:
        stock = pd.read_csv(resolve_data_path(STOCK_FILE), parse_dates=["Fecha"])
    else:
        stock["Fecha"] = pd.to_datetime(stock["Fecha"])
        
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
        st.markdown("**Tabla 1: Comparativa de Modelos de Clasificación Clínica**")
        st.caption(f"Modelo en produccion: **{risk_selected}** (seleccionado por mayor F1 en Alto Riesgo)")
        display = risk_metrics.copy()
        for col in ["Accuracy", "F1 macro", "F1 alto riesgo"]:
            display[col] = display[col].map(format_percent)
        st.dataframe(display, width="stretch", hide_index=True)
        st.caption("Descripción: Evaluación comparativa de algoritmos de clasificación de riesgo. Se prioriza la métrica F1 para la clase de Alto Riesgo con el objetivo de minimizar falsos negativos en pacientes críticos.")
    with right:
        st.markdown("**Tabla 2: Comparativa de Modelos de Regresión de Inventario**")
        st.caption(f"Modelo en produccion: **{stock_selected}** (seleccionado por menor MAE)")
        display = stock_metrics.copy()
        display["MAE"] = display["MAE"].map(lambda x: f"{x:.2f}")
        display["RMSE"] = display["RMSE"].map(lambda x: f"{x:.2f}")
        display["R2"] = display["R2"].map(format_percent)
        st.dataframe(display, width="stretch", hide_index=True)
        st.caption("Descripción: Evaluación de algoritmos de regresión para pronóstico de demanda de insumos. Se prioriza la métrica MAE por su interpretabilidad directa en unidades físicas de stock.")

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
        st.markdown("**Figure 1: Semáforo Operativo de Cobertura de Stock**")
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
        st.caption("Descripción: Días de cobertura restante por insumo crítico. Los colores indican alertas de riesgo (Rojo: Crítico <= 2.2 días, Naranja: Preventivo <= 5.6 días, Verde: Normal).")
    with col_b:
        st.markdown("**Figure 2: Distribución de Pacientes por Nivel de Prioridad de Atención**")
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
        st.caption("Descripción: Proporción de niños en el albergue según el nivel de riesgo/prioridad estimado por el modelo de ML (Random Forest/XGBoost).")

    render_model_metrics(risk_metrics, stock_metrics, risk_selected, stock_selected)
    st.info("Control de calidad: se excluye Overall_Risk_Score del modelo clinico para evitar data leakage.")


def render_inventory_view(
    stock: pd.DataFrame,
    stock_models: dict[str, object],
    stock_features: list[str],
    stock_selected: str,
) -> None:
    st.subheader("Inventario predictivo")
    
    # Alertas activas por ML
    alerts = check_active_alerts(stock, stock_models[stock_selected], stock_features)
    if alerts:
        st.error("🚨 **Alertas Activas de Stock Crítico (ML)**")
        for alert in alerts:
            if alert["severidad"] == "Crítico":
                st.markdown(f"- {alert['mensaje']}")
            else:
                st.warning(f"- {alert['mensaje']}")
    else:
        st.success("✅ **Sin Alertas de Stock:** Todos los insumos tienen niveles estables predichos por ML para los próximos 7 y 14 días.")
    
    st.divider()
    
    c1, c2, c3 = st.columns([1, 1, 1])
    selected_item = c1.selectbox("Insumo", sorted(stock["ID_Insumo"].unique()))
    horizon = c2.radio("Horizonte", ["7 dias", "14 dias"], horizontal=True)
    alert_filter = c3.multiselect(
        "Alertas", ["Critico", "Preventivo", "Normal"], default=["Critico", "Preventivo", "Normal"]
    )

    item_df = stock[stock["ID_Insumo"] == selected_item].sort_values("Fecha")
    latest = item_df.iloc[-1]
    projected_col = "Stock_Proyectado_7d" if horizon == "7 dias" else "Stock_Proyectado_14d"
    horizon_days = 7 if horizon == "7 dias" else 14

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stock actual", f"{latest['Stock_Actual']:.0f}")
    m2.metric("Ratio stock", f"{latest['Ratio_Stock']:.2f}")
    m3.metric("Punto reorden", f"{latest['Punto_Reorden']:.0f}")
    m4.metric("Alerta actual", latest["Alerta"])

    if require_plotly():
        # Calcular proyección ML
        pred_ml = predict_horizon_stock(item_df, horizon_days, stock_models[stock_selected], stock_features)
        
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
                name=f"Proyeccion lineal {horizon}",
                line=dict(color="#c1121f", dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=item_df["Fecha"],
                y=pred_ml,
                name=f"Proyeccion ML {horizon}",
                line=dict(color="#e63946", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=item_df["Fecha"],
                y=item_df["Punto_Reorden"],
                name="Punto reorden",
                line=dict(color="#f48c06", dash="dot"),
            )
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Unidades")
        st.markdown("**Figure 3: Tendencia Temporal e Inventario Predictivo del Insumo Seleccionado**")
        st.plotly_chart(fig, width="stretch")
        st.caption("Descripción: Evolución temporal del stock real frente al punto de reorden y las proyecciones lineales simples junto al pronóstico inteligente del modelo de Machine Learning (XGBoost/RF).")

    st.subheader("Lista priorizada de reposicion")
    st.markdown("**Table 3: Lista de Reposición Prioritaria de Insumos Críticos**")
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
    st.caption("Descripción: Reporte consolidado de stock actual, cobertura calculada en días de consumo promedio móvil y proyección de abastecimiento para 7 y 14 días. Las alertas están basadas en el ratio del stock actual.")
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

        st.markdown("---")
        st.markdown("#### Sustento Operativo: Mitigación del impacto de la expansión (50 a 100 familias)")
        st.markdown(
            """
            La transición hacia **ALDIMI 2.0** implica duplicar la capacidad de atención (de 50 a 100 familias).
            Desde una perspectiva cuantitativa, esto se traduce en un incremento potencial del **100% en la demanda diaria** de alimentos y medicamentos.
            
            **¿Cómo absorbe el sistema este impacto sin caer en desabastecimiento?**
            1. **Punto de Reorden Dinámico:** El punto de reorden (`Punto_Reorden = Consumo_Diario * Lead_Time`) se calcula diariamente. A medida que la ocupación se duplica, el consumo diario promedio sube, elevando automáticamente el umbral de reorden y alertando con mayor anticipación para realizar las compras.
            2. **Predicción Preventiva:** Las proyecciones a 7 y 14 días basadas en Machine Learning permiten al equipo coordinar donaciones y compras con proveedores **antes** de que el stock físico caiga a niveles críticos, absorbiendo el lead time de reposición de 14 días.
            3. **Simulación de Escenarios:** Con el simulador anterior, la administración puede modelar el ingreso de 100 familias y prever el incremento de consumo para planificar los presupuestos de adquisición de stock con base científica.
            """
        )


def render_clinical_view(
    health: pd.DataFrame,
    risk_models: dict[str, object],
    risk_features: list[str],
    label_encoder: LabelEncoder,
    confusion_df: pd.DataFrame,
    risk_selected: str,
) -> None:
    st.subheader("Priorizacion preventiva")
    
    # Identificar pacientes de alto riesgo para el disparador de alertas
    high_risk_patients = health[health["Risk_Level"].astype(str) == "High"]
    num_high_risk = len(high_risk_patients)
    
    if num_high_risk > 0:
        st.error(f"🚨 **Disparador de Alertas de Alto Riesgo Clínico-Social ({num_high_risk} Pacientes Detectados)**")
        st.markdown(
            "Los siguientes pacientes combinan factores de alto riesgo clínico (como historial familiar o mutaciones genéticas) y socioeconómicos. Se recomienda priorización médica y asignación preferente de recursos."
        )
        with st.expander("Ver lista detallada de pacientes críticos", expanded=False):
            st.dataframe(
                high_risk_patients[
                    [
                        "Patient_ID",
                        "Cancer_Type",
                        "Age",
                        "Family_History",
                        "county_CTYNAME",
                        "Diet_Risk_Index",
                        "Balance_Riesgo",
                    ]
                ].sort_values("Balance_Riesgo", ascending=False),
                width="stretch",
                hide_index=True,
            )
    else:
        st.success("✅ **Sin Alertas de Alto Riesgo:** No se han detectado pacientes clínicos críticos en prioridad Alta.")
        
    st.divider()

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
            st.markdown("**Figure 4: Pacientes por Nivel de Prioridad Filtrados**")
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
            st.caption("Descripción: Distribución de pacientes según el nivel de prioridad asignado en el conjunto de datos filtrado.")
        with right:
            st.markdown("**Figure 5: Dispersión del Perfil de Riesgo Clínico-Social**")
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
            st.caption("Descripción: Relación entre hábitos de riesgo y el balance total clínico-social, mapeado al color de prioridad asignado por el modelo.")

    st.subheader("Pacientes sugeridos para seguimiento")
    st.markdown("**Table 4: Lista Viva de Pacientes para Seguimiento y Priorización Médica**")
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
    st.caption("Descripción: Lista en tiempo real de pacientes ordenados por índice de balance de riesgo clínico-social para enfocar los esfuerzos del personal del albergue.")
    download_csv_button(follow_up, "seguimiento_clinico.csv")

    with st.expander("Matriz de confusion del modelo (validacion)"):
        st.markdown("**Figure 6: Matriz de Confusión del Modelo de Priorización**")
        model_names = list(risk_models.keys())
        model_name = st.selectbox(
            "Modelo para matriz",
            model_names,
            index=model_names.index(risk_selected) if risk_selected in model_names else 0,
            key="cm_model",
        )
        render_confusion_heatmap(confusion_df, model_name)
        st.caption("Descripción: Matriz de confusión que muestra las coincidencias de clasificación entre los datos reales y predichos del modelo seleccionado.")

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
                st.markdown("**Figure 7: Distribución de Probabilidades de Prioridad para el Paciente Simulado**")
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
                st.caption("Descripción: Gráfico de probabilidad estimado para cada una de las prioridades del paciente simulado.")


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
            st.markdown("**Figure 8: Distribución y Análisis de Densidad de la Variable Clínica**")
            fig = px.histogram(
                health,
                x=variable,
                color=health["Risk_Level"].astype(str).map(RISK_LABELS),
                color_discrete_map=RISK_COLORS,
                marginal="box",
            )
            fig.update_layout(height=480, margin=dict(l=10, r=10, t=20, b=10), bargap=0.05)
            st.plotly_chart(fig, width="stretch")
            st.caption("Descripción: Histograma de frecuencia y diagrama de caja marginal que muestra la distribución de la variable clínica seleccionada, estratificada por prioridad.")
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
            st.markdown("**Figure 9: Matriz de Correlación Numérica de Factores de Riesgo**")
            fig = px.imshow(health[cols].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
            fig.update_layout(height=620, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Descripción: Mapa de calor de coeficientes de correlación de Pearson entre las variables clínicas y socioeconómicas del dataset.")
        elif stat_view == "Dispersion":
            x_var = st.selectbox("Eje X", ["Age", "BMI", "Habitos_Riesgo", "Balance_Riesgo", "Risk_Lifestyle_Score"])
            y_var = st.selectbox("Eje Y", ["Factor_Protector", "Riesgo_Clinico", "Diet_Risk_Index", "BMI"])
            st.markdown("**Figure 10: Relación de Dispersión entre Variables del Paciente**")
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
            st.caption("Descripción: Gráfico de dispersión bidimensional que cruza dos indicadores de riesgo y resalta la prioridad asignada.")
        elif stat_view == "Outliers (Z-score)":
            variable = st.selectbox(
                "Variable numerica",
                ["Age", "BMI", "Balance_Riesgo", "Habitos_Riesgo", "Factor_Protector", "Diet_Risk_Index"],
            )
            threshold = st.slider("Umbral |Z|", 2.0, 4.0, 3.0, 0.1)
            outliers = compute_zscore_outliers(health[variable], threshold)
            st.metric("Outliers detectados", len(outliers))
            st.markdown("**Table 5: Identificación de Valores Atípicos Clínicos (Z-score > Umbral)**")
            st.dataframe(outliers, width="stretch", hide_index=True)
            st.caption(f"Descripción: Lista de registros con un puntaje Z absoluto mayor a {threshold} para la variable clínica analizada.")
            
            st.markdown("**Figure 11: Distribución y Puntos Atípicos en Diagrama de Caja**")
            fig = px.box(
                health,
                y=variable,
                color=health["Risk_Level"].astype(str).map(RISK_LABELS),
                color_discrete_map=RISK_COLORS,
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Descripción: Diagrama de caja estratificado que identifica la distribución de cuartiles y outliers de la variable clínica.")
        elif stat_view == "Resumen descriptivo":
            st.markdown("**Table 6: Resumen Estadístico Descriptivo del Dataset de Priorización Clínica**")
            st.dataframe(health.describe(include="all").transpose(), width="stretch")
            st.caption("Descripción: Métricas descriptivas agregadas (media, desviación estándar, mínimos, cuartiles y máximos) para el dataset clínico.")
            download_csv_button(health.describe(include="all").transpose().reset_index(), "resumen_clinico.csv")
        else:
            view = st.radio("Categoria", ["Historial familiar vs riesgo", "Top tipos de cancer"], horizontal=True)
            if view == "Historial familiar vs riesgo":
                st.markdown("**Figure 12: Distribución de Pacientes según Historial Familiar y Nivel de Prioridad**")
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
                fig.update_layout(height=460, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, width="stretch")
                st.caption("Descripción: Comparativa por grupos del volumen de pacientes según antecedentes familiares de cáncer y su prioridad clínica.")
            else:
                st.markdown("**Figure 13: Prevalencia de Tipos de Cáncer en el Albergue**")
                top = health["Cancer_Type"].value_counts().head(8).reset_index()
                top.columns = ["Cancer_Type", "Pacientes"]
                fig = px.bar(top, x="Cancer_Type", y="Pacientes", color_discrete_sequence=["#1d3557"])
                fig.update_layout(height=460, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, width="stretch")
                st.caption("Descripción: Frecuencia absoluta de los tipos de cáncer más comunes registrados entre los pacientes del albergue.")
    else:
        if stat_view == "Distribuciones":
            item = st.selectbox("Insumo", sorted(stock["ID_Insumo"].unique()))
            variable = st.selectbox(
                "Variable",
                ["Consumo_Diario", "Stock_Actual", "Ratio_Stock", "Punto_Reorden", "Cobertura_Dias"],
            )
            st.markdown("**Figure 14: Distribución y Densidad de Variables del Insumo**")
            fig = px.histogram(
                stock[stock["ID_Insumo"] == item],
                x=variable,
                nbins=35,
                color_discrete_sequence=["#1d3557"],
                marginal="box",
            )
            fig.update_layout(height=480, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Descripción: Histograma de frecuencia y boxplot para analizar la variabilidad del comportamiento logístico del insumo seleccionado.")
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
            st.markdown("**Figure 15: Matriz de Correlación de Pearson para Variables del Inventario**")
            fig = px.imshow(stock[cols].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
            fig.update_layout(height=620, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Descripción: Relación lineal entre variables de stock, tasas de consumo diario/móvil y variables contextuales del albergue.")
        elif stat_view == "Dispersion":
            st.markdown("**Figure 16: Dispersión entre Consumo Diario y Stock Actual**")
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
            st.caption("Descripción: Gráfico de dispersión cruzando las unidades vendidas/consumidas diariamente con los niveles de stock físico.")
        elif stat_view == "Outliers (Z-score)":
            variable = st.selectbox(
                "Variable", ["Stock_Actual", "Consumo_Diario", "Ratio_Stock", "Cobertura_Dias", "Lead_Time"]
            )
            threshold = st.slider("Umbral |Z|", 2.0, 4.0, 3.0, 0.1, key="stock_zscore")
            outliers = compute_zscore_outliers(stock[variable], threshold)
            st.metric("Outliers detectados", len(outliers))
            
            st.markdown("**Table 7: Identificación de Valores Atípicos en Inventario (Z-score)**")
            st.dataframe(outliers, width="stretch", hide_index=True)
            st.caption(f"Descripción: Lista de registros correspondientes a consumos o stocks atípicos que superan el umbral Z de {threshold}.")
            
            st.markdown("**Figure 17: Análisis de Outliers por Insumo (Diagrama de Caja)**")
            fig = px.box(stock.dropna(subset=[variable]), x="ID_Insumo", y=variable)
            fig.update_layout(height=460, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Descripción: Caja de variabilidad por tipo de insumo logístico para identificar anomalías de abastecimiento o picos de demanda.")
        elif stat_view == "Resumen descriptivo":
            st.markdown("**Table 8: Resumen Estadístico Descriptivo del Dataset de Inventario**")
            st.dataframe(stock.describe(include="all").transpose(), width="stretch")
            st.caption("Descripción: Resumen de métricas centrales y de dispersión para las variables y registros históricos de inventario.")
            download_csv_button(stock.describe(include="all").transpose().reset_index(), "resumen_inventario.csv")
        else:
            st.markdown("**Figure 18: Comparativa de Ratio de Stock por Insumo Crítico**")
            current = latest_stock(stock)
            fig = px.bar(
                current, x="ID_Insumo", y="Ratio_Stock", color="Alerta", color_discrete_map=ALERT_COLORS
            )
            fig.update_layout(height=460, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Descripción: Ratios de stock en tiempo real por tipo de insumo para evaluar la cobertura actual frente a los puntos críticos de abastecimiento.")


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

    st.markdown("#### Figure 19: Arquitectura de Datos y Flujo de Ingestion MLOps")
    st.code(
        """
[OCR / Formulario] ──> [db_infrastructure.py Ingestión] ──> [SQLite: aldimi.db]
                                                                  |
                                                      ┌───────────┴───────────┐
                                                      │                       │
                                            [Tabla: pacientes]       [Tabla: inventario]
                                                      │                       │
                                            [Modelo Clasificación]   [Modelo Regresión]
                                            (XGBoost / RF)          (XGBoost / RF)
                                                      │                       │
                                            [predicciones_riesgo]    [predicciones_stock]
                                                      │                       │
                                                      └───────────┬───────────┘
                                                                  │
                                                           [Dashboard]
                                                      ┌───────────┼───────────┐
                                                      │           │           │
                                                 [Stock]    [Triaje]    [ODS KPIs]
        """,
        language="text",
    )
    st.caption("Descripción: Diagrama de flujo del ciclo de vida del dato. Muestra desde la ingesta en tiempo real simulando un OCR hasta la consulta en caliente de las predicciones en el dashboard.")

    st.markdown("**Table 9: Arquitectura Tecnológica y Capas del Sistema MLOps**")
    architecture = pd.DataFrame(
        [
            ["Capa de datos", "Ingestion y versionado de CSV enriquecidos + SQLite Relacional", "aldimi.db (" + HEALTH_FILE + ", " + STOCK_FILE + ")"],
            ["Capa analitica", "Feature engineering automatizado por Patient_ID", "Habitos_Riesgo, Ratio_Stock, Balance_Riesgo"],
            ["Capa de modelos", "Clasificacion y regresion supervisada con re-predicciones activas", "scikit-learn, xgboost (Random Forest y XGBoost)"],
            ["Capa de servicio", "Entrenamiento en cache y prediccion en caliente (tiempo real)", "streamlit_app.py + db_infrastructure.py"],
            ["Capa de presentacion", "Visualizacion interactiva, simuladores y consolas", "Streamlit + Plotly"],
            ["Capa de gobierno", "Metricas, trazabilidad, control de leakage y etica", "tablas de evaluacion en dashboard"],
        ],
        columns=["Capa", "Funcion", "Artefacto"],
    )
    st.dataframe(architecture, width="stretch", hide_index=True)
    st.caption("Descripción: Capas del ecosistema de MLOps del proyecto ALDIMI 2.0 y su propósito de negocio.")

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
    
    # Calcular métricas ODS cuantitativas en base al sistema actual
    meds_safe_pct = (current_stock["Cobertura_Dias"] > 14.0).mean() * 100
    median_pop = health["county_POPESTIMATE2015"].median()
    high_risk_subset = health[health["Risk_Level"].astype(str) == "High"]
    vulnerable_detected = (high_risk_subset[high_risk_subset["county_POPESTIMATE2015"] < median_pop].shape[0] / max(high_risk_subset.shape[0], 1)) * 100

    st.markdown("### KPIs Ejecutivos de Impacto Social (Alineación ODS)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ODS 3: Medicinas Seguras", f"{meds_safe_pct:.1f}%", "Insumos con cobertura >14d")
    c2.metric("ODS 10: Inclusión Equitativa", f"{vulnerable_detected:.1f}%", "Casos priorizados en áreas rurales/pequeñas")
    c3.metric("Mejora en Rupturas*", "-28.5%", "Optimización de stock")
    c4.metric("Detección de Alto Riesgo", f"{high_risk_rate:.1%}", "Pacientes de prioridad alta")

    st.caption("*Estimaciones de mejora y precisión basadas en Chirinos & Vereau (2025).")

    st.markdown("#### Table 10: Matriz de Impacto Social y Alineación ODS")
    impact = pd.DataFrame(
        [
            [
                "ODS 3 - Salud",
                "Deteccion temprana de pacientes prioritarios",
                f"{high_risk_rate:.1%} del dataset en alto riesgo identificable de forma automatizada.",
            ],
            [
                "ODS 10 - Equidad",
                "Perfil 360 con variables de condado y demografía",
                f"Detección del {vulnerable_detected:.1f}% de casos críticos en condados de menor población (vulnerables).",
            ],
            [
                "Logistica",
                "Alertas preventivas de inventario",
                f"{critical_rate:.1%} de insumos en alerta critica en el estado actual.",
            ],
            [
                "Eficiencia",
                "Automatizacion de analisis manual",
                "Monitoreo constante sin intervención o errores de registro manual.",
            ],
            [
                "Seguridad alimentaria",
                "Control de insumos perecederos",
                "Métricas dinámicas a 7/14 días que reducen desperdicio y mermas.",
            ],
        ],
        columns=["Dimension ODS", "Beneficio", "Indicador / evidencia"],
    )
    st.dataframe(impact, width="stretch", hide_index=True)
    st.caption("Descripción: Matriz de impacto logístico y clínico alineada con las metas específicas de los Objetivos de Desarrollo Sostenible (ODS 3 y ODS 10).")

    st.markdown("#### Table 11: Matriz de Riesgos Éticos, Criticidad y Mitigaciones")
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
    st.caption("Descripción: Análisis y gestión de riesgos éticos asociados al uso de algoritmos predictivos y el procesamiento de datos clínicos e inventario del albergue.")

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

    st.markdown("**Table 12: Flujo de Ingestion del Ecosistema ALDIMI Core AI**")
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
    st.caption("Descripción: Flujo de integración de los entregables del proyecto, desde la adquisición de datos hasta el consumo interactivo final.")

    st.markdown("#### Integracion entre equipos")
    st.markdown("**Table 13: Roles e Integración de Actores en el Ecosistema**")
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
    st.caption("Descripción: Interacción de componentes humanos, metodológicos y computacionales para garantizar la entrega de valor.")

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


def render_sidebar(health_df, stock_df, risk_models, risk_features, label_encoder, stock_models, stock_features) -> str:
    st.sidebar.title("ALDIMI Core AI")
    st.sidebar.caption("Ecosistema de gestion inteligente")

    group = st.sidebar.selectbox("Area", list(NAV_GROUPS.keys()))
    section = st.sidebar.radio("Seccion", NAV_GROUPS[group])
    
    st.sidebar.divider()
    
    # Ingestión OCR en tiempo real (simulada)
    with st.sidebar.expander("📥 Ingestión en Tiempo Real (OCR)", expanded=False):
        ingestion_type = st.radio("Tipo de dato", ["Paciente (Clínico)", "Inventario (Logística)"], key="ingest_type")
        
        if ingestion_type == "Paciente (Clínico)":
            st.caption("Simula la digitalización OCR de la ficha del paciente.")
            with st.form("ocr_patient_form", clear_on_submit=True):
                pat_id = st.number_input("Patient ID", min_value=10000, max_value=99999, value=25000, step=1)
                cancer_type = st.selectbox("Tipo de Cáncer", ["Lung", "Colon", "Skin", "Breast", "Leukemia", "Lymphoma", "Stomach", "Kidney", "Bladder", "Prostate"])
                age = st.slider("Edad (Años)", 1, 100, 12)
                gender = st.selectbox("Género", [0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
                bmi = st.number_input("BMI (Masa Corporal)", min_value=10.0, max_value=50.0, value=20.5, step=0.1)
                
                # Factores Clínicos y Riesgos
                family_hist = st.checkbox("Historial Familiar de Cáncer", value=False)
                brca = st.checkbox("Mutación BRCA", value=False)
                hpylori = st.checkbox("Infección H. Pylori", value=False)
                
                # Hábitos
                smoking = st.slider("Consumo Tabaco (0-10)", 0, 10, 0)
                alcohol = st.slider("Consumo Alcohol (0-10)", 0, 10, 0)
                obesity = st.slider("Índice Obesidad (0-10)", 0, 10, 2)
                air_pollution = st.slider("Contaminación Aire (0-10)", 0, 10, 3)
                occ_hazards = st.slider("Riesgo Ocupacional (0-10)", 0, 10, 0)
                
                # Factores Protectores y Dieta
                physical_activity = st.slider("Actividad Física (0-10)", 0, 10, 6)
                calcium = st.slider("Consumo Calcio (0-10)", 0, 10, 5)
                red_meat = st.slider("Consumo Carne Roja (0-10)", 0, 10, 3)
                salted = st.slider("Consumo Procesados (0-10)", 0, 10, 2)
                fruit = st.slider("Consumo Fruta/Verdura (0-10)", 0, 10, 7)
                
                # Condado
                county_name = st.text_input("Nombre de Condado", value="Lima")
                county_state = st.number_input("Estado ID (county_STATE)", min_value=1, max_value=100, value=15)
                county_pop = st.number_input("Población Condado", min_value=1000, value=50000)
                
                submitted = st.form_submit_button("Ingestar y Predecir")
                
                if submitted:
                    # Enriquecimiento de variables en tiempo real
                    habitos = smoking + alcohol + obesity + air_pollution + occ_hazards
                    riesgo_clinico = int(family_hist) + int(brca) + int(hpylori)
                    factor_protector = fruit + physical_activity + calcium
                    balance = habitos + riesgo_clinico - factor_protector
                    
                    edad_rango = pd.cut(
                        [age],
                        bins=[0, 30, 45, 60, 120],
                        labels=["Joven", "Adulto", "Mayor", "Adulto_Mayor"],
                    )[0]
                    
                    patient_dict = {
                        "Patient_ID": int(pat_id),
                        "Cancer_Type": cancer_type,
                        "Age": int(age),
                        "Gender": int(gender),
                        "Smoking": int(smoking),
                        "Alcohol_Use": int(alcohol),
                        "Obesity": int(obesity),
                        "Family_History": int(family_hist),
                        "Diet_Red_Meat": int(red_meat),
                        "Diet_Salted_Processed": int(salted),
                        "Fruit_Veg_Intake": int(fruit),
                        "Physical_Activity": int(physical_activity),
                        "Air_Pollution": int(air_pollution),
                        "Occupational_Hazards": int(occ_hazards),
                        "BRCA_Mutation": int(brca),
                        "H_Pylori_Infection": int(hpylori),
                        "Calcium_Intake": int(calcium),
                        "Overall_Risk_Score": 0.0,
                        "BMI": float(bmi),
                        "Physical_Activity_Level": int(physical_activity),
                        "county_STATE": int(county_state),
                        "county_CTYNAME": county_name,
                        "county_POPESTIMATE2015": float(county_pop),
                        "Habitos_Riesgo": int(habitos),
                        "Riesgo_Clinico": int(riesgo_clinico),
                        "Factor_Protector": int(factor_protector),
                        "Balance_Riesgo": int(balance),
                        "Edad_Rango": str(edad_rango)
                    }
                    
                    # Generar predicción con el modelo preferido
                    pref_risk_model = preferred_model(risk_models)
                    input_df = pd.DataFrame([patient_dict])[risk_features]
                    
                    try:
                        pred_id = int(risk_models[pref_risk_model].predict(input_df)[0])
                        proba = risk_models[pref_risk_model].predict_proba(input_df)[0]
                        pred_label = label_encoder.inverse_transform([pred_id])[0]
                        patient_dict["Risk_Level"] = pred_label
                        
                        # Guardar en SQLite
                        import db_infrastructure as db
                        db.insert_paciente(str(DB_PATH), patient_dict)
                        db.save_prediction_riesgo(str(DB_PATH), int(pat_id), pred_label, tuple(proba))
                        
                        # Limpiar cache y recargar
                        st.cache_data.clear()
                        st.success(f"¡Paciente {pat_id} ingresado! Nivel de prioridad predicho: {RISK_LABELS.get(pred_label, pred_label)}")
                    except Exception as ex:
                        st.error(f"Error al generar predicción: {ex}")
                        
        else:  # Inventario
            st.caption("Simula el registro de consumo o stock en tiempo real.")
            with st.form("ocr_stock_form", clear_on_submit=True):
                insumo_id = st.selectbox("Insumo", sorted(stock_df["ID_Insumo"].unique()))
                fecha_val = st.date_input("Fecha", value=pd.Timestamp.now())
                stock_act = st.number_input("Stock Actual", min_value=0.0, value=500.0, step=1.0)
                consumo_d = st.number_input("Consumo Diario Reciente", min_value=0.0, value=20.0, step=1.0)
                lead_t = st.number_input("Lead Time (Días de Reposición)", min_value=1, value=14, step=1)
                
                # Variables del contexto del albergue
                ocupacion_alb = st.slider("Ocupación Albergue (0.0 - 1.0)", 0.0, 1.0, 0.7, 0.05)
                pacientes_alto = st.number_input("Pacientes de Alto Riesgo actuales", min_value=0, value=5, step=1)
                ocupacion_tot = st.number_input("Ocupación Total Familias", min_value=0, value=75, step=1)
                
                submitted = st.form_submit_button("Ingestar y Proyectar")
                
                if submitted:
                    punto_re = consumo_d * lead_t
                    ratio_s = stock_act / max(punto_re, 0.1)
                    
                    stock_dict = {
                        "Fecha": str(fecha_val),
                        "ID_Insumo": insumo_id,
                        "Stock_Actual": float(stock_act),
                        "Consumo_Diario": float(consumo_d),
                        "Lead_Time": int(lead_t),
                        "Ocupacion_Albergue": float(ocupacion_alb),
                        "Pacientes_Alto_Riesgo": int(pacientes_alto),
                        "Ocupacion_Total": int(ocupacion_tot),
                        "Punto_Reorden": float(punto_re),
                        "Ratio_Stock": float(ratio_s),
                        "Necesita_Reabastecimiento": 2 if ratio_s <= RATIO_CRITICO else (1 if ratio_s <= RATIO_PREVENTIVO else 0)
                    }
                    
                    # Ejecutar predicción
                    pref_stock_model = preferred_model(stock_models)
                    # Necesitamos calcular promedios móviles para la fila
                    item_df = stock_df[stock_df["ID_Insumo"] == insumo_id].sort_values("Fecha")
                    consumo_7d_est = float(item_df["Consumo_Diario"].tail(7).mean()) if not item_df.empty else consumo_d
                    consumo_14d_est = float(item_df["Consumo_Diario"].tail(14).mean()) if not item_df.empty else consumo_d
                    
                    stock_row = {
                        "Consumo_Diario": consumo_d,
                        "Lead_Time": lead_t,
                        "Ocupacion_Albergue": ocupacion_alb,
                        "Consumo_7d": consumo_7d_est,
                        "Consumo_14d": consumo_14d_est,
                        "Punto_Reorden": punto_re,
                        "Ratio_Stock": ratio_s,
                        "Pacientes_Alto_Riesgo": pacientes_alto,
                        "Ocupacion_Total": ocupacion_tot
                    }
                    
                    try:
                        input_df = pd.DataFrame([stock_row])[stock_features]
                        pred_stock_val = stock_models[pref_stock_model].predict(input_df)[0]
                        pred_ratio = pred_stock_val / max(punto_re, 0.1)
                        alerta_pred = alert_from_ratio(pred_ratio)
                        
                        # Guardar en SQLite
                        import db_infrastructure as db
                        db.insert_inventario(str(DB_PATH), stock_dict)
                        db.save_prediction_stock(str(DB_PATH), insumo_id, str(fecha_val), "7/14d", float(pred_stock_val), alerta_pred)
                        
                        # Limpiar cache y recargar
                        st.cache_data.clear()
                        st.success(f"¡Inventario de {insumo_id} actualizado! Alerta proyectada: {alerta_pred}")
                    except Exception as ex:
                        st.error(f"Error al generar predicción de inventario: {ex}")

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

section = render_sidebar(health_df, stock_df, risk_models, risk_features, label_encoder, stock_models, stock_features)

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
