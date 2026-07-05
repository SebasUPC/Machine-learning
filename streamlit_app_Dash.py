from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ModuleNotFoundError:  # pragma: no cover
    px = None
    go = None

try:
    import joblib
except ModuleNotFoundError:  # pragma: no cover
    joblib = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except ModuleNotFoundError:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None

from sklearn import __version__ as SKLEARN_VERSION
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))
import aldimi_common as ac  # noqa: E402
import db_infrastructure as db  # noqa: E402

DATA_PROCESSED = ac.DATA_PROCESSED
MODELS_DIR = ac.MODELS_DIR
DB_PATH = DATA_PROCESSED / ac.DB_FILE
HEALTH_FILE = ac.HEALTH_PROCESSED_FILE
STOCK_FILE = ac.STOCK_PROCESSED_FILE

RISK_ORDER = ac.PRIORITY_ORDER               # ["Bajo", "Medio", "Alto"]
RISK_COLORS = ac.PRIORITY_COLORS
ALERT_COLORS = ac.ALERT_COLORS
REABASTECIMIENTO_ALERT = {0: "Normal", 1: "Preventivo", 2: "Critico"}

# Inicializar la base de datos SQLite (ingesta desde los datasets preparados)
try:
    def _csv(nombre):
        p = DATA_PROCESSED / nombre
        return str(p) if p.exists() else ""

    db.init_db(str(DB_PATH), _csv(HEALTH_FILE), _csv(STOCK_FILE))
except Exception as e:  # pragma: no cover
    st.error(f"Error al inicializar la base de datos SQLite: {e}")

NAV_GROUPS = {
    "Operacion diaria": ["Resumen ejecutivo", "Inventario predictivo", "Priorizacion clinica"],
    "Analisis de datos": ["Menu estadistico"],
    "Estrategia y gobierno": ["MLOps", "Impacto ODS y etica", "Ecosistema Core AI"],
}

SECTION_DESCRIPTIONS = {
    "Resumen ejecutivo": "Vista unificada con KPIs, semaforos y comparacion de modelos.",
    "Inventario predictivo": "Alertas de stock, proyecciones 7/14 dias y simulador de compras.",
    "Priorizacion clinica": "Clasificacion de prioridad, seguimiento sugerido y simulador de paciente.",
    "Menu estadistico": "Distribuciones, correlaciones, outliers y estadisticas descriptivas.",
    "MLOps": "Arquitectura tecnica, pipeline de datos y despliegue del sistema.",
    "Impacto ODS y etica": "Impacto social estimado, riesgos eticos y mitigaciones.",
    "Ecosistema Core AI": "Integracion con la base comun de datos del proyecto.",
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding: 1.2rem 1.8rem 2rem 1.8rem; max-width: 1400px; }
        header[data-testid="stHeader"] {
            background: transparent !important;
            box-shadow: none !important;
        }
        div[data-testid="stToolbar"] { display: none !important; }
        section[data-testid="stSidebar"] { display: none !important; }
        .top-bar {
            background: linear-gradient(90deg, #176f93 0%, #1d9aa9 100%);
            padding: 0.8rem 1rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            color: white;
            display: flex;
            align-items: center;
            justify-content: space-between;
            width: 100%;
            box-sizing: border-box;
        }
        .top-bar h1 { margin: 0; font-size: 1.4rem; letter-spacing: 0.05em; }
        .hero-card, .page-card {
            background: white;
            border-radius: 20px;
            padding: 1.3rem;
            box-shadow: 0 14px 35px rgba(0,0,0,0.06);
        }
        .hero-card { text-align: center; margin-bottom: 1.4rem; }
        .hero-title { margin-top: 0.85rem; margin-bottom: 0.4rem; font-size: 2rem; color: #0b2947; }
        .hero-subtitle { color: #3f5d78; font-size: 1rem; margin-bottom: 1.4rem; }
        .metric-card {
            background: #eaf6fb;
            border-radius: 18px;
            padding: 1rem 1.2rem;
        }
        .metric-card h3 { margin: 0 0 0.2rem 0; font-size: 1.2rem; color: #176f93; }
        .metric-card p { margin: 0; font-size: 1.7rem; font-weight: 700; color: #0c2f44; }
        .feature-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #eef2f7 100%) !important;
            border-left: 4px solid #1d3557 !important;
            border-radius: 8px !important;
            padding: 0.85rem 1rem !important;
            margin-bottom: 0.5rem !important;
        }
        .feature-card h4 { margin: 0 0 0.25rem 0 !important; color: #1d3557 !important; font-size: 0.95rem !important; }
        .feature-card p { margin: 0 !important; color: #495057 !important; font-size: 0.85rem !important; }
        div[data-testid="stMetric"] {
            background-color: #f8f9fa !important;
            border: 1px solid #dee2e6 !important;
            border-radius: 10px !important;
            padding: 0.5rem 0.75rem !important;
        }
        div[data-testid="stMetric"] label[data-testid="stMetricLabel"] { color: #495057 !important; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #1d3557 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="ALDIMI Core AI", page_icon="A", layout="wide", initial_sidebar_state="expanded")
inject_styles()


def render_page_header(section: str) -> None:
    st.markdown(
        f"""
        <div class="top-bar">
            <h1>ALDIMI Core AI</h1>
            <div>{section}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="hero-card">
            <h2 class="hero-title">{section}</h2>
            <p class="hero-subtitle">{SECTION_DESCRIPTIONS[section]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Carga de datos (desde la base de datos comun)                               #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    health = db.fetch_all_pacientes(str(DB_PATH))
    stock = db.fetch_all_inventario(str(DB_PATH))
    if health.empty and (DATA_PROCESSED / HEALTH_FILE).exists():
        health = pd.read_csv(DATA_PROCESSED / HEALTH_FILE)
    if stock.empty and (DATA_PROCESSED / STOCK_FILE).exists():
        stock = pd.read_csv(DATA_PROCESSED / STOCK_FILE)

    health["Prioridad_Atencion"] = pd.Categorical(
        health["Prioridad_Atencion"].astype(str), categories=RISK_ORDER, ordered=True
    )
    if "Fecha" in stock.columns:
        stock["Fecha"] = pd.to_datetime(stock["Fecha"])
        stock = stock.sort_values(["ID_Insumo", "Fecha"])
    # Recalcular features derivadas si faltan (robustez)
    if "Cobertura_Dias" not in stock.columns:
        stock = ac.add_stock_features(stock)
    return health, stock


def latest_stock(stock: pd.DataFrame) -> pd.DataFrame:
    return stock.groupby("ID_Insumo", as_index=False).tail(1).sort_values("Cobertura_Dias")


# --------------------------------------------------------------------------- #
# Servicio de modelos: carga joblib (Colab) o fallback ligero en la app       #
# --------------------------------------------------------------------------- #
def _make_clf_pipeline(X, model):
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    prep = ColumnTransformer(
        [("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)]
    )
    return Pipeline([("prep", prep), ("clf", model)])


def _load_first_joblib(candidates: list[Path], label: str) -> tuple[object | None, Path | None, list[str]]:
    """Carga el primer joblib valido y devuelve errores de diagnostico."""
    errors: list[str] = []
    if joblib is None:
        return None, None, [f"{label}: joblib no esta instalado."]
    for path in candidates:
        if path is None or not path.exists():
            continue
        try:
            return joblib.load(path), path, errors
        except Exception as ex:
            errors.append(f"{label} ({path.name}): {ex}")
    return None, None, errors


@st.cache_resource(show_spinner=False)
def get_clinical_service(_health: pd.DataFrame) -> dict:
    """Devuelve modelos de clasificacion (RF y XGB), metricas y matriz de confusion.

    Prioriza los artefactos .joblib generados en Colab; si no existen, entrena
    modelos ligeros en la app (fallback de demostracion).
    """
    features = ac.health_feature_columns(_health)
    X_full = _health[features]
    le = LabelEncoder()
    y_full = le.fit_transform(_health["Prioridad_Atencion"].astype(str))
    Xtr, Xte, ytr, yte = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)

    bundle_path = ac.find_model_artifact([ac.CLF_BUNDLE_ARTIFACT])
    rf_path = ac.find_model_artifact([ac.CLF_RF_ARTIFACT])
    models: dict[str, object] = {}
    source = "fallback"
    load_error: str | None = None

    # Los clasificadores fueron serializados en Colab con scikit-learn 1.6.x.
    # En versiones nuevas cambia una clase interna de ColumnTransformer y el
    # unpickle falla; en ese caso saltamos directo al fallback para no romper UI.
    can_load_colab_classifiers = SKLEARN_VERSION.startswith("1.6.")

    if joblib is not None and can_load_colab_classifiers:
        rf_model, _, rf_errors = _load_first_joblib([rf_path] if rf_path is not None else [], "Random Forest")
        if rf_model is not None:
            models["Random Forest"] = rf_model
            source = "joblib (Colab)"
        elif rf_errors:
            load_error = " ".join(rf_errors)

        xgb_path = ac.find_model_artifact([ac.CLF_XGB_ARTIFACT])
        xgb_model, _, xgb_errors = _load_first_joblib([xgb_path] if xgb_path is not None else [], "XGBoost")
        if xgb_model is not None:
            models["XGBoost"] = xgb_model
            source = "joblib (Colab)"
        elif xgb_errors:
            load_error = f"{load_error + ' ' if load_error else ''}{' '.join(xgb_errors)}".strip()

        if bundle_path is not None:
            try:
                b = joblib.load(bundle_path)
                if isinstance(b, dict):
                    le = b.get("label_encoder", le)
                    features = b.get("features", features)
            except Exception as ex:
                load_error = (
                    f"{load_error + ' ' if load_error else ''}No se pudo cargar el bundle de clasificación ({ex})."
                ).strip()

    if not models:  # Fallback ligero
        muestra = _health.sample(min(15000, len(_health)), random_state=42)
        Xs, ys = muestra[features], le.transform(muestra["Prioridad_Atencion"].astype(str))
        Xstr, _, ystr, _ = train_test_split(Xs, ys, test_size=0.2, random_state=42, stratify=ys)
        models["Random Forest"] = _make_clf_pipeline(
            X_full, RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1)
        ).fit(Xstr, ystr)
        if XGBClassifier is not None:
            models["XGBoost"] = _make_clf_pipeline(
                X_full, XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.1, subsample=0.9,
                                      eval_metric="mlogloss", tree_method="hist", random_state=42)
            ).fit(Xstr, ystr)

    # Metricas y matrices de confusion
    idx_alto = list(le.classes_).index("Alto") if "Alto" in list(le.classes_) else len(le.classes_) - 1
    rows, confusion_rows = [], []
    for name, model in models.items():
        pred = model.predict(Xte)
        rows.append({
            "Modelo": name,
            "Accuracy": accuracy_score(yte, pred),
            "F1 macro": f1_score(yte, pred, average="macro"),
            "F1 alto riesgo": f1_score(yte == idx_alto, pred == idx_alto),
            "Recall alto": recall_score(yte == idx_alto, pred == idx_alto),
            "Falsos negativos alto": int(((yte == idx_alto) & (pred != idx_alto)).sum()),
        })
        cm = confusion_matrix(yte, pred, labels=list(range(len(le.classes_))))
        for i, real in enumerate(le.classes_):
            for j, prd in enumerate(le.classes_):
                confusion_rows.append({"Modelo": name, "Real": real, "Predicho": prd, "Casos": int(cm[i, j])})

    metrics = pd.DataFrame(rows).sort_values(["F1 alto riesgo", "F1 macro", "Accuracy"], ascending=False)
    selected = metrics.iloc[0]["Modelo"]
    return {
        "models": models, "metrics": metrics, "features": features, "label_encoder": le,
        "confusion": pd.DataFrame(confusion_rows), "selected": selected, "source": source,
        "load_error": load_error,
    }


@st.cache_resource(show_spinner=False)
def get_stock_service(_stock: pd.DataFrame) -> dict:
    """Modelos de regresion para t+7 y t+14 (joblib de Colab o fallback ligero)."""
    features = ac.stock_feature_columns(_stock)
    targets = {"t+7": ac.DEMAND_TARGET_7, "t+14": ac.DEMAND_TARGET_14}
    models: dict[str, object] = {}
    metric_rows = []
    source = "fallback"
    load_error: str | None = None

    path_t7 = ac.find_model_artifact(ac.REGRESSION_MODEL_CANDIDATES["t+7"])
    path_t14 = ac.find_model_artifact(ac.REGRESSION_MODEL_CANDIDATES["t+14"])

    if joblib is not None:
        if path_t7 is not None:
            model_t7, _, errors_t7 = _load_first_joblib(
                [MODELS_DIR / name for name in ac.REGRESSION_MODEL_CANDIDATES["t+7"]],
                "modelo t+7",
            )
            if model_t7 is not None:
                models["t+7"] = model_t7
                source = "joblib (Colab)"
            elif errors_t7:
                load_error = " ".join(errors_t7)
        if path_t14 is not None:
            model_t14, _, errors_t14 = _load_first_joblib(
                [MODELS_DIR / name for name in ac.REGRESSION_MODEL_CANDIDATES["t+14"]],
                "modelo t+14",
            )
            if model_t14 is not None:
                models["t+14"] = model_t14
                source = "joblib (Colab)"
            elif errors_t14:
                load_error = f"{load_error + ' ' if load_error else ''}{' '.join(errors_t14)}".strip()

        if all(horizon in models for horizon in targets):
            if (MODELS_DIR / "metricas_regresion.csv").exists():
                metrics = pd.read_csv(MODELS_DIR / "metricas_regresion.csv")
            else:
                metrics = pd.DataFrame()
            return {
                "models": models, "metrics": metrics, "features": features,
                "targets": targets, "source": source, "load_error": load_error,
            }

    # Fallback ligero: RF y XGB por horizonte, se conserva el mejor por MAE
    source = "fallback"
    for etiqueta, tgt in targets.items():
        d = _stock.dropna(subset=[tgt]).sort_values("Fecha")
        X = d[features].fillna(0)
        y = d[tgt]
        corte = int(len(d) * 0.8)
        Xtr, Xte, ytr, yte = X.iloc[:corte], X.iloc[corte:], y.iloc[:corte], y.iloc[corte:]
        candidatos = {"Random Forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=3, random_state=42, n_jobs=-1)}
        if XGBRegressor is not None:
            candidatos["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.9,
                                                 tree_method="hist", random_state=42)
        mejor, mejor_mae = None, np.inf
        for nombre, base in candidatos.items():
            pipe = Pipeline([("sc", StandardScaler()), ("reg", base)]).fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            mae = mean_absolute_error(yte, pred)
            metric_rows.append({"Horizonte": etiqueta, "Modelo": nombre, "MAE": round(mae, 3),
                                "RMSE": round(np.sqrt(mean_squared_error(yte, pred)), 3),
                                "R2": round(r2_score(yte, pred), 4)})
            if mae < mejor_mae:
                mejor, mejor_mae = pipe, mae
        models[etiqueta] = mejor
    return {
        "models": models, "metrics": pd.DataFrame(metric_rows), "features": features,
        "targets": targets, "source": source, "load_error": load_error,
    }


def predict_prioridad(service: dict, model_name: str, row_df: pd.DataFrame) -> tuple[str, np.ndarray]:
    model = service["models"][model_name]
    le = service["label_encoder"]
    X = row_df[service["features"]]
    pred_id = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    orden = {c: 0.0 for c in RISK_ORDER}
    for cls, p in zip(le.classes_, proba):
        orden[str(cls)] = float(p)
    return le.inverse_transform([pred_id])[0], np.array([orden[c] for c in RISK_ORDER])


def predict_demand_horizon(service: dict, horizon: str, feat_row: pd.DataFrame) -> float:
    """Predice la DEMANDA (consumo) acumulada del insumo en el horizonte dado."""
    model = service["models"][horizon]
    X = feat_row[service["features"]].fillna(0)
    return float(max(0.0, model.predict(X)[0]))


def project_stock_horizon(service: dict, horizon: str, feat_row: pd.DataFrame) -> tuple[float, float]:
    """Deriva (stock_proyectado, demanda_predicha) para el horizonte dado."""
    demanda = predict_demand_horizon(service, horizon, feat_row)
    stock_actual = float(feat_row["Stock_Actual"].iloc[0])
    return float(ac.project_stock(stock_actual, demanda)), demanda


def require_plotly() -> bool:
    if px is None or go is None:
        st.error("Falta Plotly. Instala dependencias con: pip install -r requirements.txt")
        return False
    return True


def format_percent(v: float) -> str:
    return f"{v:.2%}"


def download_csv_button(df: pd.DataFrame, filename: str, label: str = "Descargar CSV") -> None:
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    st.download_button(label, buffer.getvalue(), file_name=filename, mime="text/csv")


# --------------------------------------------------------------------------- #
# Componentes de metricas de modelos                                          #
# --------------------------------------------------------------------------- #
def render_feature_cards() -> None:
    cards = [
        ("Priorizacion clinica", "Clasifica pacientes en Bajo, Medio y Alto con simulador interactivo."),
        ("Inventario predictivo", "Anticipa quiebres con alertas Critico / Preventivo / Normal a 7 y 14 dias."),
        ("Menu estadistico", "Explora distribuciones, correlaciones y outliers sin salir del dashboard."),
    ]
    cols = st.columns(3)
    for col, (title, text) in zip(cols, cards):
        col.markdown(f'<div class="feature-card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)


def render_global_kpis(health: pd.DataFrame, stock: pd.DataFrame) -> None:
    current = latest_stock(stock)
    high = int((health["Prioridad_Atencion"].astype(str) == "Alto").sum())
    crit = int((current["Alerta"] == "Critico").sum())
    prev = int((current["Alerta"] == "Preventivo").sum())
    cov = current["Cobertura_Dias"].replace([np.inf, -np.inf], np.nan).dropna()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pacientes", f"{len(health):,}")
    c2.metric("Prioridad alta", f"{high:,}")
    c3.metric("Alertas stock", crit + prev, f"{crit} criticas")
    c4.metric("Cobertura media", f"{cov.mean():.1f} dias" if len(cov) else "N/D")


def render_model_metrics(clin: dict, stock_srv: dict) -> None:
    if clin.get("load_error"):
        st.warning(clin["load_error"])
    if stock_srv.get("load_error"):
        st.warning(stock_srv["load_error"])
    st.subheader("Evaluacion comparativa de modelos")
    left, right = st.columns(2)
    with left:
        st.markdown("**Tabla 1: Comparativa de Modelos de Clasificacion Clinica**")
        source_label = "modelos cargados desde joblib" if clin['source'] == "joblib (Colab)" else "modelo ligero de respaldo"
        st.caption(f"Fuente: {clin['source']}. Modelo en produccion: **{clin['selected']}** "
                   f"(seleccionado por mayor F1 en clase Alto).")
        st.info(f"Los resultados mostrados provienen de {source_label}.")
        disp = clin["metrics"].copy()
        for col in ["Accuracy", "F1 macro", "F1 alto riesgo", "Recall alto"]:
            if col in disp.columns:
                disp[col] = disp[col].map(format_percent)
        st.dataframe(disp, width="stretch", hide_index=True)
        st.caption("Descripcion: se prioriza el F1/Recall de la clase Alto para minimizar "
                   "falsos negativos en pacientes criticos. Conclusion: gana el modelo con mejor "
                   "deteccion de la clase Alto sin sacrificar el equilibrio global.")
    with right:
        st.markdown("**Tabla 2: Comparativa de Modelos de Regresion de Demanda de Insumos**")
        stock_source_label = "modelos cargados desde joblib" if stock_srv['source'] == "joblib (Colab)" else "modelo ligero de respaldo"
        st.caption(f"Fuente: {stock_srv['source']}. Metrica critica: MAE (unidades de consumo a reponer).")
        st.info(f"Los resultados mostrados provienen de {stock_source_label}.")
        if not stock_srv["metrics"].empty:
            disp2 = stock_srv["metrics"].copy()
            st.dataframe(disp2, width="stretch", hide_index=True)
        else:
            st.info("Metricas de regresion disponibles tras el entrenamiento en Colab.")
        st.caption("Descripcion: se predice la demanda futura (senal predecible) y de ella se deriva el "
                   "stock proyectado. Se prioriza el MAE; para cada horizonte se elige el modelo de menor MAE.")
    if XGBClassifier is None or XGBRegressor is None:
        st.warning("XGBoost no esta instalado. Instala con: pip install xgboost para la comparacion completa.")


def render_confusion_heatmap(confusion_df: pd.DataFrame, model_name: str) -> None:
    if not require_plotly():
        return
    subset = confusion_df[confusion_df["Modelo"] == model_name]
    pivot = subset.pivot(index="Real", columns="Predicho", values="Casos").reindex(index=RISK_ORDER, columns=RISK_ORDER).fillna(0)
    fig = px.imshow(pivot.values, x=pivot.columns, y=pivot.index, text_auto=True,
                    color_continuous_scale="Blues", labels={"color": "Casos"})
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# VISTA: Resumen ejecutivo                                                    #
# --------------------------------------------------------------------------- #
def render_executive_view(health, stock, clin, stock_srv) -> None:
    render_feature_cards()
    render_global_kpis(health, stock)

    with st.expander("Guia rapida para directivos", expanded=False):
        st.markdown(
            """
            1. **Revise el semaforo operativo** para identificar insumos criticos y la distribucion de prioridad.
            2. **Use Inventario predictivo** para planificar reposiciones a 7 o 14 dias.
            3. **Use Priorizacion clinica** para filtrar pacientes de prioridad alta; el modelo es apoyo, no diagnostico.
            """
        )

    if not require_plotly():
        return

    st.subheader("Semaforo operativo")
    col_a, col_b = st.columns([1.2, 1])
    current = latest_stock(stock)
    with col_a:
        st.markdown("**Figura 1: Semaforo Operativo de Cobertura de Stock**")
        data = current.head(15)
        fig = px.bar(data, x="Cobertura_Dias", y="Insumo", color="Alerta", orientation="h",
                     color_discrete_map=ALERT_COLORS,
                     labels={"Cobertura_Dias": "Dias de cobertura", "Insumo": "Insumo"})
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10), yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")
        st.caption("Descripcion: dias de cobertura restante por insumo. Rojo=Critico, Naranja=Preventivo, Verde=Normal.")
    with col_b:
        st.markdown("**Figura 2: Distribucion de Pacientes por Nivel de Prioridad de Atencion**")
        counts = health["Prioridad_Atencion"].astype(str).value_counts().reindex(RISK_ORDER)
        fig = px.pie(values=counts.values, names=counts.index, hole=0.52, color=counts.index,
                     color_discrete_map=RISK_COLORS)
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Descripcion: proporcion de pacientes segun el nivel de prioridad estimado (Bajo/Medio/Alto).")

    render_model_metrics(clin, stock_srv)
    st.info("El modelo utiliza variables clinicas y de triage disponibles en admision; la priorizacion final requiere validacion del equipo medico.")


# --------------------------------------------------------------------------- #
# VISTA: Inventario predictivo                                                #
# --------------------------------------------------------------------------- #
def check_active_alerts(stock, stock_srv) -> list[dict]:
    latest = latest_stock(stock)
    alerts = []
    for _, row in latest.iterrows():
        rowdf = pd.DataFrame([row])
        for h, dias in [("t+7", 7), ("t+14", 14)]:
            stock_proy, demanda = project_stock_horizon(stock_srv, h, rowdf)
            if stock_proy <= 0:
                alerts.append({"sev": "Critico",
                               "msg": f"**CRITICO ({row['Insumo']}):** la demanda predicha ({demanda:.0f}) agota el stock en {dias} dias (stock proyectado: {stock_proy:.0f})."})
            elif stock_proy <= row["Punto_Reorden"]:
                alerts.append({"sev": "Preventivo",
                               "msg": f"**PREVENTIVO ({row['Insumo']}):** el stock proyectado cae bajo el punto de reorden en {dias} dias ({stock_proy:.0f} vs {row['Punto_Reorden']:.0f}; demanda {demanda:.0f})."})
    return alerts


def render_inventory_view(stock, stock_srv) -> None:
    st.subheader("Inventario predictivo")
    alerts = check_active_alerts(stock, stock_srv)
    if alerts:
        st.error("Alertas Activas de Stock (Machine Learning)")
        for a in alerts[:12]:
            (st.markdown if a["sev"] == "Critico" else st.warning)(f"- {a['msg']}")
    else:
        st.success("Sin alertas: todos los insumos tienen niveles estables predichos para 7 y 14 dias.")
    st.divider()

    c1, c2, c3 = st.columns(3)
    insumos = sorted(stock["Insumo"].unique())
    sel_insumo = c1.selectbox("Insumo", insumos)
    horizon = c2.radio("Horizonte", ["7 dias", "14 dias"], horizontal=True)
    alert_filter = c3.multiselect("Alertas", ["Critico", "Preventivo", "Normal"],
                                  default=["Critico", "Preventivo", "Normal"])

    item = stock[stock["Insumo"] == sel_insumo].sort_values("Fecha")
    last = item.iloc[-1]
    hkey = "t+7" if horizon == "7 dias" else "t+14"
    dias = 7 if horizon == "7 dias" else 14

    with st.expander("Simulador de compras y cobertura", expanded=True):
        st.caption(
            "Ajuste el escenario what-if. La ocupacion actualiza en bloque "
            "`Ocupacion_Total` y `Pacientes_Alto_Riesgo` y escala el consumo para el modelo."
        )
        s1, s2, s3, s4 = st.columns(4)
        consumo = s1.number_input("Consumo diario (referencia plena ocupacion)", min_value=0.0,
                                  value=float(last["Consumo_Diario"]), step=1.0, key="sim_consumo")
        lead = s2.number_input("Lead time", min_value=1, value=int(last["Lead_Time"]), step=1, key="sim_lead")
        stock_act = s3.number_input("Stock actual", min_value=0.0, value=float(last["Stock_Actual"]),
                                    step=1.0, key="sim_stock")
        ocupacion = s4.slider("Ocupacion albergue", 0.0, 1.0, float(last.get("Ocupacion_Albergue", 0.7)),
                              0.01, key="sim_ocupacion")

        feat_sim = ac.build_stock_scenario_row(
            last, consumo_diario=consumo, lead_time=lead, stock_actual=stock_act,
            ocupacion_albergue=ocupacion,
        )
        pred_stock, demanda_pred = project_stock_horizon(stock_srv, hkey, pd.DataFrame([feat_sim]))
        alerta_sim = ac.demand_alert(pred_stock, feat_sim["Punto_Reorden"])
        occ_ctx = ac.occupancy_from_slider(ocupacion)
        consumo_ml = feat_sim["Consumo_Diario"]

        st.markdown("**Escenario simulado**")
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Demanda estimada", f"{demanda_pred:.0f}")
        r2.metric("Stock proyectado", f"{pred_stock:.0f}")
        r3.metric("Punto reorden", f"{feat_sim['Punto_Reorden']:.0f}")
        r4.metric("Ratio stock", f"{feat_sim['Ratio_Stock']:.2f}")
        r5.metric("Alerta", alerta_sim)

        if alerta_sim == "Critico":
            st.error("Accion recomendada: iniciar reposicion inmediata antes del lead time.")
        elif alerta_sim == "Preventivo":
            st.warning("Accion recomendada: planificar pedido en los proximos dias.")
        else:
            st.success("Nivel de cobertura dentro de parametros operativos.")

        st.caption(
            f"Contexto ocupacion: {occ_ctx['Ocupacion_Total']} familias | "
            f"{occ_ctx['Pacientes_Alto_Riesgo']} pacientes alto riesgo | "
            f"consumo efectivo ML: {consumo_ml:.1f} u/d (escala 0.5-1.0 x ocupacion)"
        )

        st.markdown("---")
        st.markdown("#### Sustento Operativo: Mitigacion del impacto de la expansion (50 a 100 familias)")
        st.markdown(
            """
            La transicion hacia **ALDIMI 2.0** duplica la capacidad (de 50 a 100 familias), con un
            incremento potencial del **100% en la demanda diaria** de alimentos y medicamentos.

            **Como absorbe el sistema este impacto sin desabastecerse?**
            1. **Punto de reorden dinamico:** `Punto_Reorden = Consumo_Diario x Lead_Time` sube con la
               ocupacion, alertando con mas anticipacion.
            2. **Prediccion preventiva:** las proyecciones a 7 y 14 dias basadas en ML permiten coordinar
               donaciones y compras **antes** de caer a niveles criticos.
            3. **Simulacion de escenarios:** el simulador permite modelar 100 familias y planificar
               presupuestos de adquisicion con base cientifica.
            """
        )

    st.markdown("**Ultimo registro almacenado (BD)**")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Stock actual", f"{last['Stock_Actual']:.0f}")
    b2.metric("Ratio stock", f"{last['Ratio_Stock']:.2f}")
    b3.metric("Punto reorden", f"{last['Punto_Reorden']:.0f}")
    b4.metric("Alerta actual", last["Alerta"])
    if "Fecha" in last.index:
        st.caption(f"Fecha del ultimo registro: {last['Fecha']}. "
                   "El simulador arriba proyecta un escenario what-if; los valores de BD solo cambian tras ingesta OCR.")

    if require_plotly():
        pred_val, demanda_val = pred_stock, demanda_pred
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=item["Fecha"], y=item["Stock_Actual"], name="Stock historico", line=dict(color="#1d3557")))
        fig.add_trace(go.Scatter(x=item["Fecha"], y=item["Punto_Reorden"], name="Punto reorden (BD)", line=dict(color="#f48c06", dash="dot")))
        fecha_fut = item["Fecha"].max() + pd.Timedelta(days=dias)
        fig.add_trace(go.Scatter(
            x=[item["Fecha"].max(), fecha_fut],
            y=[float(feat_sim["Stock_Actual"]), pred_val],
            name=f"Stock proyectado simulado ({horizon})",
            line=dict(color="#e63946", width=2, dash="dash"),
        ))
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Unidades")
        st.markdown("**Figura 3: Tendencia Temporal e Inventario Predictivo del Insumo Seleccionado**")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"Descripcion: historico desde la BD; linea roja = proyeccion del **escenario simulado** "
            f"a {dias} dias (demanda ML {demanda_val:.0f} -> stock proyectado {pred_val:.0f})."
        )

    st.subheader("Lista priorizada de reposicion")
    st.markdown("**Tabla 3: Lista de Reposicion Prioritaria de Insumos Criticos**")
    table = latest_stock(stock)
    table = table[table["Alerta"].isin(alert_filter)]
    cols_show = ["Insumo", "Categoria_Insumo", "Stock_Actual", "Consumo_7d", "Lead_Time",
                 "Punto_Reorden", "Ratio_Stock", "Cobertura_Dias", "Alerta"]
    st.dataframe(table[[c for c in cols_show if c in table.columns]], width="stretch", hide_index=True)
    st.caption("Descripcion: stock actual, cobertura en dias y alertas por insumo, ordenado por menor cobertura.")
    download_csv_button(table, "reposicion_priorizada.csv")


# --------------------------------------------------------------------------- #
# VISTA: Priorizacion clinica                                                 #
# --------------------------------------------------------------------------- #
def render_clinical_view(health, clin) -> None:
    st.subheader("Priorizacion preventiva")
    st.caption(f"Cohorte pediatrico-juvenil (Age < {ac.HEALTH_MAX_AGE} anos), alineada con la poblacion de ALDIMI.")
    high = health[health["Prioridad_Atencion"].astype(str) == "Alto"]
    if len(high) > 0:
        st.error(f"Disparador de Alertas de Prioridad Alta ({len(high)} pacientes detectados)")
        st.markdown("Pacientes que combinan severidad clinica y vulnerabilidad social. Se recomienda "
                    "priorizacion medica y asignacion preferente de recursos.")
        with st.expander("Ver lista detallada de pacientes criticos", expanded=False):
            cols = [c for c in ["Patient_ID", "Age", "Leukemia_Status", "Severidad_Clinica",
                                "Vulnerabilidad_Social", "Socioeconomic_Status", "Country"] if c in high.columns]
            st.dataframe(high[cols].sort_values("Severidad_Clinica", ascending=False), width="stretch", hide_index=True)
    else:
        st.success("Sin alertas de prioridad alta detectadas.")
    st.divider()

    c1, c2, c3 = st.columns(3)
    niveles = c1.multiselect("Niveles", RISK_ORDER, default=RISK_ORDER)
    socios = sorted(health["Socioeconomic_Status"].dropna().unique())
    socio_f = c2.multiselect("Nivel socioeconomico", socios, default=socios)
    solo_familia = c3.toggle("Solo con historial familiar", value=False)

    filt = health[health["Prioridad_Atencion"].astype(str).isin(niveles) & health["Socioeconomic_Status"].isin(socio_f)]
    if solo_familia and "Family_History" in filt.columns:
        filt = filt[filt["Family_History"].astype(str).str.lower() == "yes"]

    if require_plotly():
        left, right = st.columns([1, 1.15])
        with left:
            st.markdown("**Figura 4: Pacientes por Nivel de Prioridad (filtrados)**")
            counts = filt["Prioridad_Atencion"].astype(str).value_counts().reindex(niveles).fillna(0)
            fig = px.bar(x=counts.index, y=counts.values, color=counts.index, color_discrete_map=RISK_COLORS,
                         labels={"x": "Prioridad", "y": "Pacientes"})
            fig.update_layout(height=390, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Descripcion: distribucion de pacientes por nivel de prioridad en el subconjunto filtrado.")
        with right:
            st.markdown("**Figura 5: Perfil Clinico (WBC vs Hemoglobina) por Prioridad**")
            muestra = filt.sample(min(4000, len(filt)), random_state=42) if len(filt) else filt
            fig = px.scatter(muestra, x="WBC_Count", y="Hemoglobin_Level",
                             color=muestra["Prioridad_Atencion"].astype(str), color_discrete_map=RISK_COLORS,
                             opacity=0.55, hover_data=["Patient_ID", "Age", "Bone_Marrow_Blasts"])
            fig.update_layout(height=390, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Descripcion: relacion entre marcadores hematologicos coloreada por la prioridad asignada.")

    st.subheader("Pacientes sugeridos para seguimiento")
    st.markdown("**Tabla 4: Lista Viva de Pacientes para Seguimiento y Priorizacion Medica**")
    follow = filt.sort_values(["Severidad_Clinica", "Vulnerabilidad_Social"], ascending=False).head(30)
    cols = [c for c in ["Patient_ID", "Age", "Edad_Rango", "Leukemia_Status", "Severidad_Clinica",
                        "Habitos_Riesgo", "Riesgo_Antecedentes", "Vulnerabilidad_Social",
                        "Socioeconomic_Status", "Prioridad_Atencion"] if c in follow.columns]
    st.dataframe(follow[cols], width="stretch", hide_index=True)
    st.caption("Descripcion: pacientes ordenados por severidad clinica y vulnerabilidad social para enfocar recursos.")
    download_csv_button(follow, "seguimiento_clinico.csv")

    with st.expander("Matriz de confusion del modelo (validacion)"):
        st.markdown("**Figura 6: Matriz de Confusion del Modelo de Priorizacion**")
        model_names = list(clin["models"].keys())
        mname = st.selectbox("Modelo para matriz", model_names,
                             index=model_names.index(clin["selected"]) if clin["selected"] in model_names else 0,
                             key="cm_model")
        render_confusion_heatmap(clin["confusion"], mname)
        st.caption("Descripcion: coincidencias entre la prioridad real y la predicha por el modelo seleccionado.")

    render_patient_simulator(health, clin)


def render_patient_simulator(health, clin) -> None:
    with st.expander("Simulador de prioridad de paciente", expanded=True):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            age = st.slider("Edad", 1, ac.HEALTH_MAX_AGE - 1, 12)
            bmi = st.slider("BMI", 12.0, 40.0, 20.0, 0.1)
            gender = st.selectbox("Genero", ["Male", "Female"])
            leukemia = st.selectbox("Diagnostico leucemia", ["Negative", "Positive"])
        with col_b:
            wbc = st.slider("WBC Count", 1000, 400000, 8000, 100)
            hb = st.slider("Hemoglobina", 5.0, 18.0, 12.0, 0.1)
            blasts = st.slider("Blastos medula (%)", 0, 100, 10)
            platelet = st.slider("Plaquetas", 10000, 450000, 200000, 1000)
        with col_c:
            genetic = st.toggle("Mutacion genetica")
            family = st.toggle("Historial familiar")
            smoking = st.toggle("Exposicion tabaco")
            socio = st.selectbox("Nivel socioeconomico", ["Low", "Medium", "High"])
            zona = st.selectbox("Zona", ["Rural", "Urban"])

        model_names = list(clin["models"].keys())
        mname = st.selectbox("Modelo de clasificacion", model_names,
                             index=model_names.index(clin["selected"]) if clin["selected"] in model_names else 0)

        # Construir fila cruda y enriquecer con las mismas funciones del pipeline
        base = health.iloc[0].to_dict()
        base.update({
            "Age": age, "BMI": bmi, "Gender": gender, "Leukemia_Status": leukemia,
            "WBC_Count": wbc, "Hemoglobin_Level": hb, "Bone_Marrow_Blasts": blasts, "Platelet_Count": platelet,
            "Genetic_Mutation": "Yes" if genetic else "No", "Family_History": "Yes" if family else "No",
            "Smoking_Status": "Yes" if smoking else "No", "Socioeconomic_Status": socio, "Urban_Rural": zona,
        })
        raw = pd.DataFrame([base])
        for col in ac.HEALTH_YESNO_COLS:
            if col in raw.columns:
                raw[col] = raw[col].astype(str)
        enr = ac.add_health_features(raw)
        label, proba = predict_prioridad(clin, mname, enr)

        r1, r2 = st.columns([0.85, 1.15])
        with r1:
            st.metric("Prioridad estimada", label)
            st.caption("Herramienta de apoyo preventivo; no reemplaza evaluacion medica.")
        with r2:
            if require_plotly():
                st.markdown("**Figura 7: Distribucion de Probabilidades de Prioridad (paciente simulado)**")
                pdf = pd.DataFrame({"Prioridad": RISK_ORDER, "Probabilidad": proba})
                fig = px.bar(pdf, x="Prioridad", y="Probabilidad", color="Prioridad", color_discrete_map=RISK_COLORS)
                fig.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=20, b=10), yaxis_tickformat=".0%")
                st.plotly_chart(fig, width="stretch")
                st.caption("Descripcion: probabilidad estimada para cada nivel de prioridad del paciente simulado.")


# --------------------------------------------------------------------------- #
# VISTA: Menu estadistico                                                     #
# --------------------------------------------------------------------------- #
def compute_zscore_outliers(series: pd.Series, threshold: float = 3.0) -> pd.DataFrame:
    clean = series.dropna()
    std = clean.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.DataFrame(columns=["Valor", "Z-score"])
    z = (clean - clean.mean()) / std
    flagged = clean[np.abs(z) > threshold]
    return pd.DataFrame({"Valor": flagged.values, "Z-score": z[np.abs(z) > threshold].values}).head(20)


def render_statistics_view(health, stock) -> None:
    st.subheader("Menu estadistico")
    st.caption("Exploracion interactiva alineada con el EDA: distribuciones, correlaciones, outliers y resumen.")
    dataset = st.radio("Dataset", ["Clinico", "Inventario"], horizontal=True)
    stat_view = st.selectbox("Tipo de analisis", ["Distribuciones", "Correlaciones", "Dispersion",
                                                   "Outliers (Z-score)", "Resumen descriptivo", "Analisis por categoria"])
    if not require_plotly():
        return

    if dataset == "Clinico":
        num_vars = ["Age", "BMI", "WBC_Count", "Hemoglobin_Level", "Bone_Marrow_Blasts", "Platelet_Count",
                    "Severidad_Clinica", "Habitos_Riesgo", "Vulnerabilidad_Social"]
        num_vars = [c for c in num_vars if c in health.columns]
        if stat_view == "Distribuciones":
            var = st.selectbox("Variable", num_vars)
            st.markdown("**Figura 8: Distribucion de la Variable Clinica**")
            fig = px.histogram(health, x=var, color=health["Prioridad_Atencion"].astype(str),
                               color_discrete_map=RISK_COLORS, marginal="box")
            fig.update_layout(height=480, margin=dict(l=10, r=10, t=20, b=10), bargap=0.05)
            st.plotly_chart(fig, width="stretch")
            st.caption("Descripcion: histograma y boxplot marginal estratificado por prioridad.")
        elif stat_view == "Correlaciones":
            st.markdown("**Figura 9: Matriz de Correlacion de Factores Clinicos**")
            fig = px.imshow(health[num_vars].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
            fig.update_layout(height=620, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Descripcion: correlacion de Pearson entre variables clinicas y socioeconomicas.")
        elif stat_view == "Dispersion":
            xv = st.selectbox("Eje X", num_vars, index=0)
            yv = st.selectbox("Eje Y", num_vars, index=min(3, len(num_vars) - 1))
            st.markdown("**Figura 10: Relacion de Dispersion entre Variables del Paciente**")
            muestra = health.sample(min(5000, len(health)), random_state=42)
            fig = px.scatter(muestra, x=xv, y=yv, color=muestra["Prioridad_Atencion"].astype(str),
                             color_discrete_map=RISK_COLORS, opacity=0.6)
            fig.update_layout(height=480, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Descripcion: dispersion bidimensional resaltando la prioridad asignada.")
        elif stat_view == "Outliers (Z-score)":
            var = st.selectbox("Variable numerica", num_vars)
            thr = st.slider("Umbral |Z|", 2.0, 4.0, 3.0, 0.1)
            out = compute_zscore_outliers(health[var], thr)
            st.metric("Outliers detectados", len(out))
            st.markdown("**Tabla 5: Identificacion de Valores Atipicos Clinicos (Z-score)**")
            st.dataframe(out, width="stretch", hide_index=True)
            st.markdown("**Figura 11: Diagrama de Caja de la Variable Clinica**")
            fig = px.box(health, y=var, color=health["Prioridad_Atencion"].astype(str), color_discrete_map=RISK_COLORS)
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Descripcion: cuartiles y outliers de {var} por nivel de prioridad.")
        elif stat_view == "Resumen descriptivo":
            st.markdown("**Tabla 6: Resumen Estadistico Descriptivo (Clinico)**")
            desc = health[num_vars].describe().T
            st.dataframe(desc, width="stretch")
            download_csv_button(desc.reset_index(), "resumen_clinico.csv")
        else:
            view = st.radio("Categoria", ["Socioeconomico vs prioridad", "Top paises"], horizontal=True)
            if view == "Socioeconomico vs prioridad":
                st.markdown("**Figura 12: Distribucion por Nivel Socioeconomico y Prioridad**")
                g = health.groupby(["Socioeconomic_Status", health["Prioridad_Atencion"].astype(str)]).size().reset_index(name="Pacientes")
                g.columns = ["Socioeconomico", "Prioridad", "Pacientes"]
                fig = px.bar(g, x="Socioeconomico", y="Pacientes", color="Prioridad", barmode="group", color_discrete_map=RISK_COLORS)
                fig.update_layout(height=460, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, width="stretch")
                st.caption("Descripcion: volumen de pacientes por nivel socioeconomico y prioridad (relevante para ODS 10).")
            else:
                st.markdown("**Figura 13: Top 8 Paises por Numero de Pacientes**")
                top = health["Country"].value_counts().head(8).reset_index()
                top.columns = ["Country", "Pacientes"]
                fig = px.bar(top, x="Country", y="Pacientes", color_discrete_sequence=["#1d3557"])
                fig.update_layout(height=460, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, width="stretch")
                st.caption("Descripcion: paises con mayor numero de pacientes registrados.")
    else:
        num_vars = ["Stock_Actual", "Consumo_Diario", "Ratio_Stock", "Punto_Reorden", "Cobertura_Dias",
                    "Consumo_7d", "Consumo_14d", "Lead_Time", "Ocupacion_Total", "Pacientes_Alto_Riesgo"]
        num_vars = [c for c in num_vars if c in stock.columns]
        if stat_view == "Distribuciones":
            insumo = st.selectbox("Insumo", sorted(stock["Insumo"].unique()))
            var = st.selectbox("Variable", num_vars)
            st.markdown("**Figura 14: Distribucion y Densidad de Variables del Insumo**")
            fig = px.histogram(stock[stock["Insumo"] == insumo], x=var, nbins=35, color_discrete_sequence=["#1d3557"], marginal="box")
            fig.update_layout(height=480, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Descripcion: variabilidad del comportamiento logistico del insumo seleccionado.")
        elif stat_view == "Correlaciones":
            st.markdown("**Figura 15: Matriz de Correlacion de Variables del Inventario**")
            fig = px.imshow(stock[num_vars].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
            fig.update_layout(height=620, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Descripcion: relacion lineal entre stock, consumo y variables contextuales.")
        elif stat_view == "Dispersion":
            st.markdown("**Figura 16: Dispersion entre Consumo Diario y Stock Actual**")
            fig = px.scatter(stock.sample(min(5000, len(stock)), random_state=42), x="Consumo_Diario", y="Stock_Actual",
                             color="Alerta", color_discrete_map=ALERT_COLORS, opacity=0.5, hover_data=["Insumo"])
            fig.update_layout(height=480, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Descripcion: relacion entre consumo diario y stock fisico por nivel de alerta.")
        elif stat_view == "Outliers (Z-score)":
            var = st.selectbox("Variable", num_vars)
            thr = st.slider("Umbral |Z|", 2.0, 4.0, 3.0, 0.1, key="stock_z")
            out = compute_zscore_outliers(stock[var], thr)
            st.metric("Outliers detectados", len(out))
            st.markdown("**Tabla 7: Identificacion de Valores Atipicos en Inventario (Z-score)**")
            st.dataframe(out, width="stretch", hide_index=True)
            st.markdown("**Figura 17: Analisis de Outliers por Categoria de Insumo**")
            fig = px.box(stock.dropna(subset=[var]), x="Categoria_Insumo", y=var)
            fig.update_layout(height=460, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Descripcion: variabilidad de {var} por categoria de insumo.")
        elif stat_view == "Resumen descriptivo":
            st.markdown("**Tabla 8: Resumen Estadistico Descriptivo (Inventario)**")
            desc = stock[num_vars].describe().T
            st.dataframe(desc, width="stretch")
            download_csv_button(desc.reset_index(), "resumen_inventario.csv")
        else:
            st.markdown("**Figura 18: Ratio de Stock Promedio por Categoria de Insumo**")
            g = latest_stock(stock).groupby("Categoria_Insumo")["Ratio_Stock"].mean().reset_index()
            fig = px.bar(g, x="Categoria_Insumo", y="Ratio_Stock", color="Categoria_Insumo")
            fig.update_layout(height=460, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Descripcion: ratio de stock actual por categoria para evaluar cobertura frente al punto de reorden.")


# --------------------------------------------------------------------------- #
# VISTA: MLOps                                                                #
# --------------------------------------------------------------------------- #
def render_mlops_view(clin, stock_srv) -> None:
    st.subheader("Implementacion y Despliegue (MLOps)")
    st.markdown("Pipeline end-to-end que conecta la **base comun de datos** con modelos supervisados y el dashboard.")
    st.markdown("#### Figura 19: Arquitectura de Datos y Flujo MLOps")
    st.code(
        """
[Kaggle / OCR / Formulario] --> [db_infrastructure.py] --> [SQLite: aldimi.db]
                                                               |
                                             +-----------------+-----------------+
                                             |                                   |
                                     [Tabla: pacientes]                 [Tabla: inventario]
                                             |                                   |
                                 [Clasificacion RF/XGBoost]        [Regresion RF/XGBoost t+7/t+14]
                                             |                                   |
                                   [predicciones_riesgo]              [predicciones_stock]
                                             |                                   |
                                             +-----------------+-----------------+
                                                               |
                                                        [Dashboard Streamlit]
        """,
        language="text",
    )
    st.caption("Descripcion: del dato crudo/ingesta a la prediccion consultada en el dashboard.")

    st.markdown("**Tabla 9: Arquitectura Tecnologica y Capas del Sistema**")
    arch = pd.DataFrame(
        [
            ["Capa de datos", "Ingesta y versionado + SQLite relacional", f"aldimi.db ({HEALTH_FILE}, {STOCK_FILE})"],
            ["Capa analitica", "Feature engineering compartido (src/aldimi_common.py)", "Severidad_Clinica, Ratio_Stock, Consumo_7d/14d"],
            ["Capa de modelos", "Clasificacion y regresion (RF vs XGBoost) con tuning", "scikit-learn, xgboost, imbalanced-learn"],
            ["Capa de servicio", "Carga de joblib (Colab) o entrenamiento en cache", "streamlit_app_Dash.py + models/*.joblib"],
            ["Capa de presentacion", "Visualizacion, simuladores y consolas", "Streamlit + Plotly"],
            ["Capa de gobierno", "Metricas, control de calidad y etica", "notebooks 09-11 + dashboard"],
        ],
        columns=["Capa", "Funcion", "Artefacto"],
    )
    st.dataframe(arch, width="stretch", hide_index=True)
    st.caption("Descripcion: capas del ecosistema MLOps de ALDIMI 2.0 y su proposito.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Stack tecnologico")
        st.markdown(
            """
            | Componente | Tecnologia |
            |---|---|
            | Lenguaje | Python 3.10+ |
            | Datos | Pandas, NumPy, SQLite |
            | ML | scikit-learn, XGBoost, imbalanced-learn |
            | UI | Streamlit |
            | Graficos | Plotly |
            | Entrenamiento | Google Colab (joblib) |
            """
        )
    with c2:
        st.markdown("#### Despliegue recomendado")
        st.markdown(
            """
            1. `pip install -r requirements.txt`
            2. Ejecutar notebooks Hito 1-2 (generan datos + BD).
            3. Entrenar modelos avanzados en Colab (Hito 3) -> `models/*.joblib`.
            4. `streamlit run streamlit_app_Dash.py`
            5. Produccion: migrar SQLite -> **MySQL/BigQuery**, contenedor Docker y autenticacion.
            """
        )

    st.markdown("#### Modelos en produccion")
    st.markdown(f"- **Clasificacion clinica:** {clin['selected']} (fuente: {clin['source']}).\n"
                f"- **Regresion de inventario:** mejor por MAE en cada horizonte (fuente: {stock_srv['source']}).")
    render_model_metrics(clin, stock_srv)
    st.info("Antes de uso real: anonimizar identificadores, versionar el modelo y establecer revision humana obligatoria en alertas clinicas.")


# --------------------------------------------------------------------------- #
# VISTA: Impacto ODS y etica                                                  #
# --------------------------------------------------------------------------- #
def render_impact_ethics_view(health, stock) -> None:
    st.subheader("Analisis de Impacto ODS y Etica de Datos")
    current = latest_stock(stock)
    critical_rate = (current["Alerta"] == "Critico").mean()
    high_rate = (health["Prioridad_Atencion"].astype(str) == "Alto").mean()
    meds_safe = (current["Cobertura_Dias"].replace([np.inf, -np.inf], np.nan) > 14.0).mean() * 100
    alto = health[health["Prioridad_Atencion"].astype(str) == "Alto"]
    vulnerables = (alto["Vulnerabilidad_Social"] >= 1).mean() * 100 if len(alto) else 0.0

    st.markdown("### KPIs Ejecutivos de Impacto Social (Alineacion ODS)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ODS 3: Insumos seguros", f"{meds_safe:.1f}%", "Cobertura >14 dias")
    c2.metric("ODS 10: Casos vulnerables", f"{vulnerables:.1f}%", "Prioridad alta con vulnerabilidad")
    c3.metric("Insumos en alerta critica", f"{critical_rate:.1%}", "Estado actual")
    c4.metric("Deteccion prioridad alta", f"{high_rate:.1%}", "Del total de pacientes")

    st.markdown("#### Tabla 10: Matriz de Impacto Social y Alineacion ODS")
    impact = pd.DataFrame(
        [
            ["ODS 3 - Salud", "Deteccion temprana de pacientes prioritarios", f"{high_rate:.1%} identificable automaticamente."],
            ["ODS 10 - Equidad", "Variables sociales (nivel socioeconomico, zona rural)", f"{vulnerables:.1f}% de casos altos con vulnerabilidad social."],
            ["Logistica", "Alertas preventivas de inventario", f"{critical_rate:.1%} de insumos en alerta critica."],
            ["Eficiencia", "Automatizacion del analisis manual", "Monitoreo continuo sin errores de registro manual."],
            ["Seguridad alimentaria", "Control de insumos perecederos", "Metricas a 7/14 dias que reducen desperdicio."],
        ],
        columns=["Dimension ODS", "Beneficio", "Indicador / evidencia"],
    )
    st.dataframe(impact, width="stretch", hide_index=True)
    st.caption("Descripcion: impacto logistico y clinico alineado con las metas de los ODS 3 y ODS 10.")

    st.markdown("#### Tabla 11: Matriz de Riesgos Eticos y Mitigaciones")
    ethics = pd.DataFrame(
        [
            ["Sesgo por datos publicos no pediatricos", "Alto", "Validar con datos reales anonimizados; no usar como diagnostico."],
            ["Falso negativo en prioridad alta", "Alto", "Priorizar recall de la clase Alto; revision humana obligatoria."],
            ["Uso inadecuado de scores automaticos", "Alto", "Revision humana obligatoria; el sistema es apoyo a la decision."],
            ["Privacidad de pacientes", "Alto", "Anonimizacion, minimo dato necesario, control de acceso."],
            ["Sobreconfianza en probabilidades", "Medio", "Mostrar incertidumbre y disclaimers en simuladores."],
            ["Desbalance de clases", "Medio", "SMOTE dentro del Pipeline + metricas macro y por clase."],
        ],
        columns=["Riesgo", "Criticidad", "Mitigacion"],
    )
    st.dataframe(ethics, width="stretch", hide_index=True)
    st.caption("Descripcion: gestion de riesgos eticos del uso de algoritmos predictivos sobre datos clinicos y logisticos.")
    st.warning("El sistema es una herramienta de apoyo. Toda alerta clinica debe ser validada por el equipo medico y social de ALDIMI.")


# --------------------------------------------------------------------------- #
# VISTA: Ecosistema Core AI                                                   #
# --------------------------------------------------------------------------- #
def render_ecosystem_view() -> None:
    st.subheader('Confluencia con el Ecosistema "ALDIMI Core AI"')
    st.markdown("ALDIMI Core AI integra el trabajo de ML con la **base comun de datos**, que el dashboard consume de forma unificada.")
    st.markdown("**Tabla 12: Flujo de Ingestion del Ecosistema**")
    flow = pd.DataFrame(
        [
            ["02_Adquisicion_Datos.ipynb", "Ingesta desde Kaggle (salud + inventario)", "data/raw/"],
            ["05/06 Preparacion", "Limpieza + feature engineering", f"{HEALTH_FILE}, {STOCK_FILE}"],
            ["07_Integracion_BD.ipynb", "Integracion relacional", "aldimi.db"],
            ["09/10 Modelado (Colab)", "RF vs XGBoost + tuning", "models/*.joblib"],
            ["streamlit_app_Dash.py", "Capa de negocio: KPIs, alertas, simuladores", "Dashboard ALDIMI Core AI"],
        ],
        columns=["Modulo", "Responsabilidad", "Salida"],
    )
    st.dataframe(flow, width="stretch", hide_index=True)
    st.caption("Descripcion: integracion de entregables, desde la adquisicion hasta el consumo interactivo final.")

    st.markdown("#### Tabla 13: Roles e Integracion de Actores")
    integ = pd.DataFrame(
        [
            ["Base comun IA", "Esquema unificado de pacientes e insumos", "Diccionario de datos compartido"],
            ["Motor ML", "Modelos entrenados y metricas", "Predicciones y alertas"],
            ["Dashboard", "Traduccion a lenguaje de negocio", "Streamlit interactivo"],
            ["Usuarios ALDIMI", "Direccion, logistica, asistencia social", "Decisiones preventivas"],
            ["Retroalimentacion", "Nuevos registros operativos", "Reentrenamiento periodico"],
        ],
        columns=["Actor / componente", "Aporte", "Resultado"],
    )
    st.dataframe(integ, width="stretch", hide_index=True)
    st.caption("Descripcion: interaccion de componentes humanos, metodologicos y computacionales.")
    st.markdown("#### Lecciones aprendidas en la integracion")
    st.markdown(
        """
        - **Feature engineering centralizado** (`src/aldimi_common.py`) garantiza coherencia entre preparacion y despliegue.
        - **Separar datos procesados** (`data/processed/`) desacopla experimentacion e interfaz.
        - **Validacion cruzada y particion estratificada** aseguran metricas robustas y reproducibles.
        - **Versionar modelos y datasets** es prerequisito antes de escalar de 50 a 100 familias.
        """
    )
    st.info("Repositorio del proyecto: https://github.com/SebasUPC/Machine-learning")


# --------------------------------------------------------------------------- #
# Navegacion principal + ingreso OCR                                         #
# --------------------------------------------------------------------------- #
def set_pending_page(page_name: str) -> None:
    st.session_state.next_page = page_name


def render_top_navigation() -> str:
    if "page" not in st.session_state:
        st.session_state.page = "Inicio"

    if "next_page" in st.session_state:
        st.session_state.page = st.session_state.pop("next_page")

    st.markdown(
        """
        <div class='top-bar'>
            <div><strong>ALDIMI Core AI</strong></div>
            <div style='color:#ffffff;font-size:0.95rem;'>Navegación principal</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page_options = [
        "Inicio",
        "Resumen ejecutivo",
        "Inventario predictivo",
        "Priorizacion clinica",
        "Menu estadistico",
        "MLOps",
        "Impacto ODS y etica",
        "Ecosistema Core AI",
    ]
    selected_index = page_options.index(st.session_state.page) if st.session_state.page in page_options else 0
    selected = st.radio(
        "Vista",
        page_options,
        index=selected_index,
        horizontal=True,
        label_visibility="collapsed",
        key="page",
    )
    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)
    return selected


def render_welcome_page() -> None:
    st.markdown(
        """
        <div class='hero-card'>
            <h2 class='hero-title'>Bienvenido a ALDIMI Core AI</h2>
            <p class='hero-subtitle'>Selecciona la vista principal para explorar inventario o priorización clínica.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.button(
            "CONTROL STOCK",
            key="btn_welcome_stock",
            on_click=set_pending_page,
            args=("Inventario predictivo",),
        )
    with c2:
        st.button(
            "GRAVEDAD PACIENTE",
            key="btn_welcome_gravedad",
            on_click=set_pending_page,
            args=("Priorizacion clinica",),
        )
    with c3:
        st.button(
            "MODELADO",
            key="btn_welcome_modelado",
            on_click=set_pending_page,
            args=("Resumen ejecutivo",),
        )
    st.info("Usa las pestañas superiores o los botones rápidos para navegar entre las vistas principales.")


def render_sidebar(health, stock, clin, stock_srv) -> str:
    page = render_top_navigation()
    with st.expander("Ingestion en Tiempo Real (OCR)", expanded=False):
        tipo = st.radio("Tipo de dato", ["Paciente (Clinico)", "Inventario (Logistica)"], key="ingest_type")
        if tipo == "Paciente (Clinico)":
            _form_paciente(health, clin)
        else:
            _form_inventario(stock, stock_srv)
    return page


def _form_paciente(health, clin) -> None:
    st.caption("Simula la digitalizacion OCR de la ficha del paciente.")
    with st.form("ocr_patient_form", clear_on_submit=True):
        pid = st.number_input("Patient ID", min_value=100000, max_value=9999999, value=200001, step=1)
        age = st.slider("Edad", 1, ac.HEALTH_MAX_AGE - 1, 12)
        gender = st.selectbox("Genero", ["Male", "Female"])
        country = st.text_input("Pais", value="Peru")
        wbc = st.number_input("WBC Count", min_value=1000, max_value=400000, value=8000, step=100)
        rbc = st.number_input("RBC Count", min_value=1.0, max_value=8.0, value=4.8, step=0.1)
        platelet = st.number_input("Plaquetas", min_value=10000, max_value=500000, value=200000, step=1000)
        hb = st.number_input("Hemoglobina", min_value=5.0, max_value=18.0, value=12.0, step=0.1)
        blasts = st.slider("Blastos medula (%)", 0, 100, 10)
        bmi = st.number_input("BMI", min_value=10.0, max_value=45.0, value=20.0, step=0.1)
        genetic = st.checkbox("Mutacion genetica")
        family = st.checkbox("Historial familiar")
        smoking = st.checkbox("Exposicion tabaco")
        alcohol = st.checkbox("Consumo alcohol")
        radiation = st.checkbox("Exposicion radiacion")
        infection = st.checkbox("Antecedente infecciones")
        chronic = st.checkbox("Enfermedad cronica")
        immune = st.checkbox("Trastorno inmunologico")
        socio = st.selectbox("Nivel socioeconomico", ["Low", "Medium", "High"])
        zona = st.selectbox("Zona", ["Rural", "Urban"])
        leukemia = st.selectbox("Diagnostico leucemia", ["Negative", "Positive"])
        submit = st.form_submit_button("Ingestar y Predecir")

        if submit:
            base = health.iloc[0].to_dict()
            base.update({
                "Patient_ID": int(pid), "Age": int(age), "Gender": gender, "Country": country,
                "WBC_Count": int(wbc), "RBC_Count": float(rbc), "Platelet_Count": int(platelet),
                "Hemoglobin_Level": float(hb), "Bone_Marrow_Blasts": int(blasts), "BMI": float(bmi),
                "Genetic_Mutation": "Yes" if genetic else "No", "Family_History": "Yes" if family else "No",
                "Smoking_Status": "Yes" if smoking else "No", "Alcohol_Consumption": "Yes" if alcohol else "No",
                "Radiation_Exposure": "Yes" if radiation else "No", "Infection_History": "Yes" if infection else "No",
                "Chronic_Illness": "Yes" if chronic else "No", "Immune_Disorders": "Yes" if immune else "No",
                "Socioeconomic_Status": socio, "Urban_Rural": zona, "Leukemia_Status": leukemia,
            })
            raw = pd.DataFrame([base])
            enr = ac.add_health_features(raw)
            try:
                label, proba = predict_prioridad(clin, clin["selected"], enr)
                registro = enr.iloc[0].to_dict()
                registro["Prioridad_Atencion"] = label
                registro = {k: (int(v) if isinstance(v, np.integer) else (float(v) if isinstance(v, np.floating) else v))
                            for k, v in registro.items()}
                db.insert_paciente(str(DB_PATH), registro)
                db.save_prediction_riesgo(str(DB_PATH), int(pid), label, tuple(proba))
                st.cache_data.clear()
                st.success(f"Paciente {pid} ingresado. Prioridad predicha: {label}")
                st.rerun()
            except Exception as ex:
                st.error(f"Error al generar prediccion: {ex}")


def _form_inventario(stock, stock_srv) -> None:
    st.caption("Simula el registro de consumo o stock en tiempo real.")
    with st.form("ocr_stock_form", clear_on_submit=True):
        insumo = st.selectbox("Insumo", sorted(stock["Insumo"].unique()))
        fecha = st.date_input("Fecha", value=pd.Timestamp.now())
        stock_act = st.number_input("Stock actual", min_value=0.0, value=500.0, step=1.0)
        consumo = st.number_input("Consumo diario", min_value=0.0, value=20.0, step=1.0)
        lead = st.number_input("Lead time (dias)", min_value=1, value=14, step=1)
        ocupacion = st.slider("Ocupacion albergue", 0.0, 1.0, 0.7, 0.05)
        submit = st.form_submit_button("Ingestar y Proyectar")

        if submit:
            item = stock[stock["Insumo"] == insumo].sort_values("Fecha")
            base = item.iloc[-1].to_dict() if len(item) else stock.iloc[-1].to_dict()
            base["Fecha"] = str(fecha)
            registro = ac.build_stock_scenario_row(
                base, consumo_diario=consumo, lead_time=lead, stock_actual=stock_act,
                ocupacion_albergue=ocupacion,
            )
            punto = registro["Punto_Reorden"]
            try:
                pred_stock, demanda_pred = project_stock_horizon(stock_srv, "t+7", pd.DataFrame([registro]))
                alerta = ac.demand_alert(pred_stock, punto)
                db_row = {k: v for k, v in registro.items() if k in _inventario_columns()}
                db_row = {k: (int(v) if isinstance(v, np.integer) else (float(v) if isinstance(v, np.floating) else v))
                            for k, v in db_row.items()}
                db.insert_inventario(str(DB_PATH), db_row)
                db.save_prediction_stock(str(DB_PATH), insumo, str(fecha), "7/14d", float(pred_stock), alerta)
                st.cache_data.clear()
                st.success(f"Inventario de {insumo} actualizado. Demanda t+7 predicha: {demanda_pred:.0f} -> "
                           f"stock proyectado {pred_stock:.0f}. Alerta: {alerta}.")
                st.rerun()
            except Exception as ex:
                st.error(f"Error al generar prediccion de inventario: {ex}")


def _inventario_columns() -> set:
    conn = db.get_connection(str(DB_PATH))
    try:
        cols = pd.read_sql_query("SELECT * FROM inventario LIMIT 1;", conn).columns.tolist()
    except Exception:
        cols = []
    conn.close()
    return set(cols)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
health_df, stock_df = load_data()
clin_service = get_clinical_service(health_df)
stock_service = get_stock_service(stock_df)

section = render_sidebar(health_df, stock_df, clin_service, stock_service)

if section == "Inicio":
    render_welcome_page()
elif section == "Resumen ejecutivo":
    render_executive_view(health_df, stock_df, clin_service, stock_service)
elif section == "Inventario predictivo":
    render_inventory_view(stock_df, stock_service)
elif section == "Priorizacion clinica":
    render_clinical_view(health_df, clin_service)
elif section == "Menu estadistico":
    render_statistics_view(health_df, stock_df)
elif section == "MLOps":
    render_mlops_view(clin_service, stock_service)
elif section == "Impacto ODS y etica":
    render_impact_ethics_view(health_df, stock_df)
elif section == "Ecosistema Core AI":
    render_ecosystem_view()
else:
    render_welcome_page()
