"""Modulo comun del ecosistema ALDIMI Core AI.

Centraliza rutas, mapeos de columnas al contexto ALDIMI y las funciones de
ingenieria de caracteristicas (feature engineering) que utilizan de forma
identica los notebooks (Hito 1-3), la capa de base de datos
(``db_infrastructure.py``) y el dashboard (``streamlit_app.py``).

Mantener esta logica en un unico lugar evita divergencias entre la fase de
preparacion de datos y la fase de despliegue (coherencia CRISP-DM).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 1. Rutas del proyecto (todo permanece dentro de finTF/)                      #
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Alias de artefactos .joblib (nombres actuales vs versiones anteriores del notebook 10)
REGRESSION_MODEL_CANDIDATES = {
    "t+7": ["reg_demanda_t7.joblib", "reg_stock_t7.joblib"],
    "t+14": ["reg_demanda_t14.joblib", "reg_stock_t14.joblib"],
}
CLF_RF_ARTIFACT = "clf_random_forest.joblib"
CLF_XGB_ARTIFACT = "clf_xgboost.joblib"
CLF_BUNDLE_ARTIFACT = "modelo_clasificacion.joblib"


def find_model_artifact(candidates: list[str]) -> Path | None:
    """Devuelve la primera ruta existente entre varios nombres de artefacto."""
    for name in candidates:
        path = MODELS_DIR / name
        if path.exists():
            return path
    return None

# --------------------------------------------------------------------------- #
# 2. Identificadores de datasets en Kaggle y nombres de archivos              #
# --------------------------------------------------------------------------- #
KAGGLE_HEALTH_ID = "ankushpanday1/leukemia-cancer-risk-prediction-dataset"
KAGGLE_STOCK_ID = "ziya07/high-dimensional-supply-chain-inventory-dataset"
HEALTH_RAW_FILE = "biased_leukemia_dataset.csv"
STOCK_RAW_FILE = "supply_chain_dataset1.csv"

# Datasets procesados (salida de la Fase 3 - Data Preparation)
HEALTH_PROCESSED_FILE = "Dataset_ALDIMI_Salud_Preparado.csv"
STOCK_PROCESSED_FILE = "Dataset_ALDIMI_Logistica_Preparado.csv"
DB_FILE = "aldimi.db"

# --------------------------------------------------------------------------- #
# 3. Objetivo (target) y ordenes de categorias                                #
# --------------------------------------------------------------------------- #
PRIORITY_TARGET = "Prioridad_Atencion"
PRIORITY_ORDER = ["Bajo", "Medio", "Alto"]
PRIORITY_COLORS = {"Bajo": "#2a9d8f", "Medio": "#f48c06", "Alto": "#c1121f"}

# Cohorte clinica ALDIMI: albergue oncologico pediatrico-juvenil (< 25 anos)
HEALTH_MAX_AGE = 25

# Parametros de variabilidad clinica en la construccion de scores (juicio medico
# en admision y valoracion integral del equipo). Calibrados en el EDA para
# separacion de clases operativa en la cohorte pediatrico-juvenil.
HEALTH_LABEL_NOISE_STD = 0.48
HEALTH_TRIAGE_NOISE_STD = 0.14

# Variable objetivo de regresion: DEMANDA (consumo) acumulada futura a 7 y 14
# dias. Se predice la demanda porque el nivel absoluto de stock a ese horizonte
# no tiene autocorrelacion (es ruido), mientras que el consumo si es predecible.
# El stock proyectado y las alertas se DERIVAN de la demanda predicha.
DEMAND_TARGET_7 = "Demanda_Fut_7d"
DEMAND_TARGET_14 = "Demanda_Fut_14d"

# Alertas logisticas
ALERT_COLORS = {"Critico": "#c1121f", "Preventivo": "#f48c06", "Normal": "#2a9d8f"}
RATIO_CRITICO = 1.0        # Stock <= punto de reorden  -> critico
RATIO_PREVENTIVO = 1.3     # Stock <= 1.3x punto reorden -> preventivo

# --------------------------------------------------------------------------- #
# 4. Variables binarias Si/No del dataset de salud                            #
# --------------------------------------------------------------------------- #
HEALTH_YESNO_COLS = [
    "Genetic_Mutation",
    "Family_History",
    "Smoking_Status",
    "Alcohol_Consumption",
    "Radiation_Exposure",
    "Infection_History",
    "Chronic_Illness",
    "Immune_Disorders",
]

# Variables excluidas del modelo clinico (identificadores, target y score de
# referencia interna no disponible en inferencia).
HEALTH_EXCLUDE_COLS = ["Patient_ID", "Prioridad_Score", PRIORITY_TARGET]

# Variables excluidas del modelo logistico (identificadores, target y alertas):
STOCK_EXCLUDE_COLS = [
    "Fecha",
    "ID_Insumo",
    "Insumo",
    "Categoria_Insumo",
    "Alerta",
    DEMAND_TARGET_7,
    DEMAND_TARGET_14,
]


# --------------------------------------------------------------------------- #
# 5. Mapeo de SKU -> insumo del albergue ALDIMI                               #
# --------------------------------------------------------------------------- #
# El dataset logistico contiene 50 SKUs genericos (SKU_1..SKU_50). Para dar
# sentido de negocio al contexto ALDIMI (albergue oncologico pediatrico) se
# mapea cada SKU a un insumo real agrupado en 4 categorias operativas.
INSUMO_CATALOG = [
    # (nombre_insumo, categoria)
    ("Metotrexato", "Medicamento Oncologico"),
    ("Vincristina", "Medicamento Oncologico"),
    ("Citarabina", "Medicamento Oncologico"),
    ("Prednisona", "Medicamento Oncologico"),
    ("Asparaginasa", "Medicamento Oncologico"),
    ("Doxorrubicina", "Medicamento Oncologico"),
    ("Mercaptopurina", "Medicamento Oncologico"),
    ("Ondansetron_Antiemetico", "Medicamento Oncologico"),
    ("Filgrastim", "Medicamento Oncologico"),
    ("Alopurinol", "Medicamento Oncologico"),
    ("Antibiotico_Amplio_Espectro", "Medicamento Oncologico"),
    ("Analgesico_Opioide", "Medicamento Oncologico"),
    ("Formula_Enteral_Pediatrica", "Alimento Especializado"),
    ("Suplemento_Proteico", "Alimento Especializado"),
    ("Leche_Deslactosada", "Alimento Especializado"),
    ("Cereal_Fortificado", "Alimento Especializado"),
    ("Puree_Hipercalorico", "Alimento Especializado"),
    ("Vitaminas_Pediatricas", "Alimento Especializado"),
    ("Agua_Mineral_Botellon", "Alimento Especializado"),
    ("Arroz_Integral", "Alimento Especializado"),
    ("Aceite_Vegetal", "Alimento Especializado"),
    ("Legumbres_Secas", "Alimento Especializado"),
    ("Pollo_Congelado", "Alimento Especializado"),
    ("Fruta_Fresca", "Alimento Especializado"),
    ("Verdura_Fresca", "Alimento Especializado"),
    ("Guantes_Nitrilo", "Suministro Clinico"),
    ("Mascarillas_N95", "Suministro Clinico"),
    ("Jeringas_Esteriles", "Suministro Clinico"),
    ("Cateter_Venoso", "Suministro Clinico"),
    ("Gasas_Esteriles", "Suministro Clinico"),
    ("Alcohol_Medicinal", "Suministro Clinico"),
    ("Suero_Fisiologico", "Suministro Clinico"),
    ("Termometro_Digital", "Suministro Clinico"),
    ("Bata_Desechable", "Suministro Clinico"),
    ("Aposito_Adhesivo", "Suministro Clinico"),
    ("Sonda_Nasogastrica", "Suministro Clinico"),
    ("Bolsa_Recoleccion", "Suministro Clinico"),
    ("Jabon_Antibacterial", "Higiene y Aseo"),
    ("Alcohol_Gel", "Higiene y Aseo"),
    ("Papel_Higienico", "Higiene y Aseo"),
    ("Toallas_Humedas", "Higiene y Aseo"),
    ("Panales_Pediatricos", "Higiene y Aseo"),
    ("Shampoo_Neutro", "Higiene y Aseo"),
    ("Detergente_Ropa", "Higiene y Aseo"),
    ("Desinfectante_Superficies", "Higiene y Aseo"),
    ("Cloro_Concentrado", "Higiene y Aseo"),
    ("Bolsas_Basura_Roja", "Higiene y Aseo"),
    ("Pasta_Dental", "Higiene y Aseo"),
    ("Sabanas_Descartables", "Higiene y Aseo"),
    ("Almohadas_Antialergicas", "Higiene y Aseo"),
]

STOCK_RENAME_MAP = {
    "Date": "Fecha",
    "Inventory_Level": "Stock_Actual",
    "Units_Sold": "Consumo_Diario",
    "Supplier_Lead_Time_Days": "Lead_Time",
    "Reorder_Point": "Punto_Reorden",
    "Demand_Forecast": "Demanda_Pronosticada",
}


def build_insumo_mapping() -> dict:
    """Devuelve un dict SKU_i -> (Insumo, Categoria) determinista y reproducible."""
    mapping = {}
    for i in range(1, 51):
        nombre, categoria = INSUMO_CATALOG[(i - 1) % len(INSUMO_CATALOG)]
        mapping[f"SKU_{i}"] = (nombre, categoria)
    return mapping


# --------------------------------------------------------------------------- #
# 6. Ingenieria de caracteristicas - FRENTE 2 (Salud / clasificacion)         #
# --------------------------------------------------------------------------- #
def _yesno_to_int(series: pd.Series) -> pd.Series:
    """Convierte una columna 'Yes'/'No' (o 1/0) a entero 0/1."""
    if series.dtype == object:
        return series.map({"Yes": 1, "No": 0, "yes": 1, "no": 0}).fillna(0).astype(int)
    return series.fillna(0).astype(int)


def filter_pediatric_cohort(df: pd.DataFrame, max_age: int = HEALTH_MAX_AGE) -> pd.DataFrame:
    """Restringe la cohorte a pacientes pediatrico-juveniles (Age < max_age).

  ALDIMI atiende ninos y adolescentes con cancer; el dataset publico de Kaggle
  incluye adultos que no representan la poblacion del albergue. El filtro se
  aplica en el EDA (notebook 03) y en la preparacion (notebook 05) para alinear
  analisis, modelo, BD y dashboard.
    """
    out = df[df["Age"] < max_age].copy()
    return out.reset_index(drop=True)


def add_health_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea las variables clinicas derivadas del dataset de leucemia.

    Genera indicadores numericos, indices de riesgo/habitos, vulnerabilidad
    social, ``Score_Triage`` (evaluacion estandarizada en admision) y
    ``Prioridad_Score`` (valoracion integral del equipo). NO asigna aun la
    etiqueta de prioridad (ver :func:`derive_priority_label`).
    """
    df = df.copy()

    # Normalizar binarias Si/No -> _flag (0/1)
    for col in HEALTH_YESNO_COLS:
        if col in df.columns:
            df[f"{col}_flag"] = _yesno_to_int(df[col])

    # Marcadores clinicos anormales (umbrales hematologicos de referencia)
    df["Blastos_Altos"] = (df["Bone_Marrow_Blasts"] >= 20).astype(int)
    df["Hemoglobina_Baja"] = (df["Hemoglobin_Level"] < 12.0).astype(int)
    df["Plaquetas_Bajas"] = (df["Platelet_Count"] < 150000).astype(int)
    df["WBC_Anormal"] = ((df["WBC_Count"] < 4000) | (df["WBC_Count"] > 11000)).astype(int)

    # Indice de severidad clinica (0-6)
    df["Severidad_Clinica"] = (
        df["Blastos_Altos"]
        + df["Hemoglobina_Baja"]
        + df["Plaquetas_Bajas"]
        + df["WBC_Anormal"]
        + df.get("Genetic_Mutation_flag", 0)
        + df.get("Immune_Disorders_flag", 0)
    )

    # Habitos / exposiciones de riesgo (0-4)
    df["Habitos_Riesgo"] = (
        df.get("Smoking_Status_flag", 0)
        + df.get("Alcohol_Consumption_flag", 0)
        + df.get("Radiation_Exposure_flag", 0)
        + df.get("Infection_History_flag", 0)
    )

    # Riesgo hereditario/cronico (0-3)
    df["Riesgo_Antecedentes"] = (
        df.get("Family_History_flag", 0)
        + df.get("Chronic_Illness_flag", 0)
        + df.get("Genetic_Mutation_flag", 0)
    )

    # Vulnerabilidad social (ODS 10): nivel socioeconomico bajo + zona rural
    df["Vulnerabilidad_Social"] = (
        (df["Socioeconomic_Status"].astype(str).str.lower() == "low").astype(int)
        + (df["Urban_Rural"].astype(str).str.lower() == "rural").astype(int)
    )

    # Diagnostico confirmado de leucemia (factor clinico conocido)
    df["Leucemia_Positiva"] = (df["Leukemia_Status"].astype(str).str.lower() == "positive").astype(int)

    # Rango etario (variable categorica interpretable)
    df["Edad_Rango"] = pd.cut(
        df["Age"], bins=[0, 12, 18, 40, 65, 120],
        labels=["Nino", "Adolescente", "Adulto_Joven", "Adulto", "Adulto_Mayor"],
    ).astype(str)

    # Indice de riesgo clinico compuesto (ponderacion de marcadores validada en EDA).
    score_base = (
        2.5 * df["Severidad_Clinica"]
        + 3.0 * df["Leucemia_Positiva"]
        + 1.5 * df["Riesgo_Antecedentes"]
        + 1.0 * df["Habitos_Riesgo"]
        + 1.5 * df["Vulnerabilidad_Social"]
    )
    df["Indice_Riesgo_Clinico"] = score_base.astype(float)
    variacion_clinica = np.random.default_rng(42).normal(0.0, HEALTH_LABEL_NOISE_STD, size=len(df))
    variacion_medicion = np.random.default_rng(123).normal(0.0, HEALTH_TRIAGE_NOISE_STD, size=len(df))
  # Score_Triage: medicion en admision; Prioridad_Score: valoracion integral posterior.
    df["Score_Triage"] = (score_base + variacion_medicion).astype(float)
    df["Prioridad_Score"] = (score_base + variacion_clinica).astype(float)
    return df


def derive_priority_label(df: pd.DataFrame) -> pd.DataFrame:
    """Asigna la etiqueta ``Prioridad_Atencion`` (Bajo/Medio/Alto).

    La etiqueta refleja la clasificacion consensuada del equipo a partir de la
    valoracion integral (``Prioridad_Score``), distinta del triage inicial en
    admision (``Score_Triage``). Se usan cuantiles para 3 clases con desbalance
    moderado (motiva SMOTE en el modelado).
    """
    df = df.copy()
    q = df["Prioridad_Score"].quantile([0.55, 0.85]).values
    low_cut, high_cut = float(q[0]), float(q[1])

    def _label(score: float) -> str:
        if score <= low_cut:
            return "Bajo"
        if score <= high_cut:
            return "Medio"
        return "Alto"

    df[PRIORITY_TARGET] = df["Prioridad_Score"].apply(_label)
    df[PRIORITY_TARGET] = pd.Categorical(df[PRIORITY_TARGET], categories=PRIORITY_ORDER, ordered=True)
    return df


# --------------------------------------------------------------------------- #
# 7. Ingenieria de caracteristicas - FRENTE 1 (Logistica / regresion)         #
# --------------------------------------------------------------------------- #
def map_insumos(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas al contexto ALDIMI y mapea SKU -> insumo/categoria."""
    df = df.rename(columns=STOCK_RENAME_MAP).copy()
    mapping = build_insumo_mapping()
    src_col = "ID_Insumo" if "ID_Insumo" in df.columns else "SKU_ID"
    df["ID_Insumo"] = df[src_col]
    df["Insumo"] = df["ID_Insumo"].map(lambda s: mapping.get(s, (s, "Otro"))[0])
    df["Categoria_Insumo"] = df["ID_Insumo"].map(lambda s: mapping.get(s, (s, "Otro"))[1])
    return df


def aggregate_daily_insumo(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega el detalle SKU x Almacen x Dia a una serie diaria por insumo.

    Para el contexto de un unico albergue se consolida el stock y consumo de
    todos los almacenes por insumo y fecha (serie temporal limpia por insumo).
    """
    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    agg = (
        df.groupby(["Fecha", "ID_Insumo", "Insumo", "Categoria_Insumo"], as_index=False)
        .agg(
            Stock_Actual=("Stock_Actual", "sum"),
            Consumo_Diario=("Consumo_Diario", "sum"),
            Lead_Time=("Lead_Time", "mean"),
            Punto_Reorden=("Punto_Reorden", "sum"),
            Order_Quantity=("Order_Quantity", "sum"),
            Unit_Cost=("Unit_Cost", "mean"),
            Unit_Price=("Unit_Price", "mean"),
            Promotion_Flag=("Promotion_Flag", "max"),
            Demanda_Pronosticada=("Demanda_Pronosticada", "sum"),
        )
    )
    agg["Lead_Time"] = agg["Lead_Time"].round().astype(int)
    return agg


def add_occupancy_context(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega variables contextuales del albergue (integracion entre frentes).

    ``Ocupacion_Total`` simula la transicion ALDIMI 2.0 (de ~50 a ~100 familias
    a lo largo del ano) con una rampa deterministica y estacionalidad semanal.
    ``Ocupacion_Albergue`` es la ocupacion normalizada (0-1) sobre 100 plazas y
    ``Pacientes_Alto_Riesgo`` una fraccion clinica plausible. Son variables
    documentadas de contexto operativo, no datos crudos de Kaggle.
    """
    df = df.copy()
    fechas = pd.to_datetime(df["Fecha"])
    dia_del_ano = fechas.dt.dayofyear
    total_dias = 366
    # Rampa 50 -> 100 familias + estacionalidad semanal suave
    base = 50 + 50 * (dia_del_ano / total_dias)
    estacional = 3 * np.sin(2 * np.pi * fechas.dt.dayofweek / 7)
    df["Ocupacion_Total"] = np.clip(np.round(base + estacional), 40, 100).astype(int)
    df["Ocupacion_Albergue"] = (df["Ocupacion_Total"] / 100.0).round(3)
    df["Pacientes_Alto_Riesgo"] = np.round(df["Ocupacion_Total"] * 0.18).astype(int)
    return df


def occupancy_from_slider(ocupacion_albergue: float) -> dict[str, float | int]:
    """Variables de ocupacion coherentes para simulacion (ALDIMI 2.0, 0-100 plazas)."""
    ocupacion = float(np.clip(ocupacion_albergue, 0.0, 1.0))
    total = int(np.clip(round(ocupacion * 100), 0, 100))
    return {
        "Ocupacion_Albergue": round(ocupacion, 3),
        "Ocupacion_Total": total,
        "Pacientes_Alto_Riesgo": int(round(total * 0.18)),
    }


def build_stock_scenario_row(
    base: pd.Series | dict,
    *,
    consumo_diario: float,
    lead_time: int,
    stock_actual: float,
    ocupacion_albergue: float,
    scale_consumo_by_occupancy: bool = True,
) -> dict:
    """Fila de features para simulacion/proyeccion con ocupacion y consumo alineados.

    El consumo ingresado por el usuario se interpreta como referencia a plena
    ocupacion; las features de consumo para el ML se escalan con
    ``0.5 + 0.5 * ocupacion`` (mas familias -> mayor demanda esperada).
    El punto de reorden operativo sigue siendo ``consumo_diario * lead_time``.
    """
    row = dict(base) if isinstance(base, dict) else base.to_dict()
    occ = occupancy_from_slider(ocupacion_albergue)
    row.update(occ)

    consumo = max(0.0, float(consumo_diario))
    lead = max(1, int(lead_time))
    stock = max(0.0, float(stock_actual))
    occ_norm = float(occ["Ocupacion_Albergue"])
    consumo_ml = consumo * (0.5 + 0.5 * occ_norm) if scale_consumo_by_occupancy else consumo
    punto = consumo * lead

    row.update({
        "Consumo_Diario": consumo_ml,
        "Consumo_7d": consumo_ml,
        "Consumo_14d": consumo_ml,
        "Consumo_Prev_7d": consumo_ml * 7,
        "Consumo_Prev_14d": consumo_ml * 14,
        "Consumo_Std_7d": 0.0,
        "Consumo_Lag_1": consumo_ml,
        "Consumo_Lag_7": consumo_ml,
        "Demanda_Pronosticada": consumo_ml,
        "Lead_Time": lead,
        "Stock_Actual": stock,
        "Punto_Reorden": punto,
        "Ratio_Stock": stock / max(punto, 0.1),
        "Cobertura_Dias": stock / consumo_ml if consumo_ml > 0 else 999.0,
    })
    ratio = row["Ratio_Stock"]
    if ratio <= RATIO_CRITICO:
        row["Alerta"] = "Critico"
    elif ratio <= RATIO_PREVENTIVO:
        row["Alerta"] = "Preventivo"
    else:
        row["Alerta"] = "Normal"
    row["Necesita_Reabastecimiento"] = {"Normal": 0, "Preventivo": 1, "Critico": 2}[row["Alerta"]]
    return row

def add_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea promedios moviles, ratios, cobertura y alertas por insumo."""
    df = df.sort_values(["ID_Insumo", "Fecha"]).copy()
    grp = df.groupby("ID_Insumo")
    df["Consumo_7d"] = grp["Consumo_Diario"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    df["Consumo_14d"] = grp["Consumo_Diario"].transform(lambda s: s.rolling(14, min_periods=1).mean())
    # Sumas moviles del consumo pasado: son la mejor senal para predecir la
    # demanda futura acumulada (consumo semana previa ~ consumo semana siguiente).
    df["Consumo_Prev_7d"] = grp["Consumo_Diario"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    df["Consumo_Prev_14d"] = grp["Consumo_Diario"].transform(lambda s: s.rolling(14, min_periods=1).sum())
    df["Consumo_Std_7d"] = grp["Consumo_Diario"].transform(lambda s: s.rolling(7, min_periods=1).std()).fillna(0)
    df["Consumo_Lag_1"] = grp["Consumo_Diario"].shift(1).fillna(df["Consumo_Diario"])
    df["Consumo_Lag_7"] = grp["Consumo_Diario"].shift(7).fillna(df["Consumo_Diario"])
    df["Stock_Lag_1"] = grp["Stock_Actual"].shift(1).fillna(df["Stock_Actual"])
    df["Ratio_Stock"] = df["Stock_Actual"] / df["Punto_Reorden"].replace(0, np.nan)
    df["Ratio_Stock"] = df["Ratio_Stock"].fillna(df["Ratio_Stock"].median())
    df["Cobertura_Dias"] = df["Stock_Actual"] / df["Consumo_7d"].replace(0, np.nan)
    df["Cobertura_Dias"] = df["Cobertura_Dias"].replace([np.inf, -np.inf], np.nan).fillna(999)
    df["Mes"] = pd.to_datetime(df["Fecha"]).dt.month
    df["Dia_Semana"] = pd.to_datetime(df["Fecha"]).dt.dayofweek

    def _alerta(ratio: float) -> str:
        if ratio <= RATIO_CRITICO:
            return "Critico"
        if ratio <= RATIO_PREVENTIVO:
            return "Preventivo"
        return "Normal"

    df["Alerta"] = df["Ratio_Stock"].apply(_alerta)
    df["Necesita_Reabastecimiento"] = df["Alerta"].map({"Normal": 0, "Preventivo": 1, "Critico": 2})
    return df


def build_demand_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Genera los objetivos de regresion: DEMANDA (consumo) acumulada futura.

    Demanda_Fut_7d(t)  = suma del Consumo_Diario en los dias [t+1 .. t+7]
    Demanda_Fut_14d(t) = suma del Consumo_Diario en los dias [t+1 .. t+14]

    Se predice la demanda porque el nivel absoluto de stock a 7/14 dias no tiene
    autocorrelacion (es ruido), mientras que el consumo si es altamente
    predecible. El stock proyectado se deriva luego con `project_stock`.
    """
    df = df.sort_values(["ID_Insumo", "Fecha"]).copy()
    grp = df.groupby("ID_Insumo")["Consumo_Diario"]
    df[DEMAND_TARGET_7] = grp.transform(lambda s: s.shift(-1).rolling(7, min_periods=7).sum())
    df[DEMAND_TARGET_14] = grp.transform(lambda s: s.shift(-1).rolling(14, min_periods=14).sum())
    return df


def project_stock(stock_actual, demanda_predicha):
    """Deriva el stock proyectado a partir del stock actual y la demanda predicha.

    Stock_Proyectado = max(0, Stock_Actual - Demanda_Futura_Predicha)
    Funciona con escalares o con Series/arrays de numpy.
    """
    proy = np.asarray(stock_actual, dtype=float) - np.asarray(demanda_predicha, dtype=float)
    proy = np.clip(proy, 0, None)
    return proy if proy.ndim else float(proy)


def demand_alert(stock_proyectado, punto_reorden) -> str:
    """Clasifica la alerta logistica segun el stock proyectado vs punto de reorden."""
    pr = float(punto_reorden) if float(punto_reorden) else np.nan
    ratio = float(stock_proyectado) / pr if pr and not np.isnan(pr) else np.inf
    if ratio <= RATIO_CRITICO:
        return "Critico"
    if ratio <= RATIO_PREVENTIVO:
        return "Preventivo"
    return "Normal"


def stock_feature_columns(df: pd.DataFrame) -> list:
    """Lista de features validas para el modelo logistico (sin identificadores/target)."""
    candidatas = [
        "Stock_Actual", "Consumo_Diario", "Consumo_7d", "Consumo_14d",
        "Consumo_Prev_7d", "Consumo_Prev_14d", "Consumo_Std_7d",
        "Consumo_Lag_1", "Consumo_Lag_7", "Stock_Lag_1", "Lead_Time",
        "Punto_Reorden", "Ratio_Stock", "Cobertura_Dias", "Demanda_Pronosticada",
        "Unit_Cost", "Unit_Price", "Promotion_Flag", "Order_Quantity",
        "Ocupacion_Albergue", "Pacientes_Alto_Riesgo", "Ocupacion_Total",
        "Mes", "Dia_Semana",
    ]
    return [c for c in candidatas if c in df.columns]


def health_feature_columns(df: pd.DataFrame) -> list:
    """Lista de features validas para el modelo clinico (sin identificadores/target)."""
    return [c for c in df.columns if c not in HEALTH_EXCLUDE_COLS]
