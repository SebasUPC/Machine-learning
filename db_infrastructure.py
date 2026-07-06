"""Capa de base de datos (SQLite) del ecosistema ALDIMI Core AI.

Implementa la **integracion de datos** (Fase 3 de CRISP-DM): los datasets
procesados de salud y logistica se persisten en una base relacional unica
(``aldimi.db``), que el dashboard y los modelos consumen en lugar de leer CSV
sueltos. La eleccion de SQLite es por portabilidad academica; el esquema es
directamente migrable a **MySQL / BigQuery** para produccion (persistencia,
trazabilidad, auditoria y escalabilidad de 50 a 100 familias).

El esquema se crea de forma dinamica a partir de los datasets preparados para
mantener coherencia con la fase de preparacion (esquema real de los datos).
"""
import os
import sqlite3

import pandas as pd


def get_connection(db_path: str):
    """Establece una conexion con la base de datos SQLite."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _table_is_empty(cursor, table: str) -> bool:
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,)
    )
    if cursor.fetchone() is None:
        return True
    cursor.execute(f"SELECT COUNT(*) FROM {table};")
    return cursor.fetchone()[0] == 0


def _table_needs_refresh(cursor, table: str, csv_path: str, required_columns: tuple[str, ...]) -> bool:
    """Retorna True si la tabla debe resembrarse desde un CSV procesado."""
    if not csv_path or not os.path.exists(csv_path):
        return False
    if _table_is_empty(cursor, table):
        return True

    try:
        csv_columns = set(pd.read_csv(csv_path).columns)
    except Exception:
        return False

    table_columns = {row[1] for row in cursor.execute(f"PRAGMA table_info({table});").fetchall()}
    return not all(column in table_columns for column in required_columns) or not all(
        column in csv_columns for column in required_columns
    )


def init_db(db_path: str, health_csv_path: str, stock_csv_path: str):
    """Crea las tablas e ingesta los datasets procesados si estan vacias.

    - ``pacientes``  <- dataset de salud preparado (esquema real de leucemia).
    - ``inventario`` <- dataset de logistica preparado (serie por insumo/dia).
    - ``predicciones_riesgo`` / ``predicciones_stock`` <- historial de inferencias.
    """
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Tablas de historial de predicciones (esquema fijo)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predicciones_riesgo (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Patient_ID INTEGER,
            Prioridad_Pred TEXT,
            Proba_Bajo REAL,
            Proba_Medio REAL,
            Proba_Alto REAL,
            Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predicciones_stock (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ID_Insumo TEXT,
            Fecha TEXT,
            Horizonte TEXT,
            Stock_Proyectado REAL,
            Alerta TEXT,
            Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()

    # Ingesta dinamica del dataset de salud preparado
    if _table_needs_refresh(cursor, "pacientes", health_csv_path, ("Patient_ID", "Prioridad_Atencion", "Prioridad_Score")):
        if health_csv_path and os.path.exists(health_csv_path):
            print(f"Sembrando tabla 'pacientes' desde {health_csv_path}...")
            pd.read_csv(health_csv_path).to_sql("pacientes", conn, if_exists="replace", index=False)
        else:
            print(f"ADVERTENCIA: dataset de salud no encontrado en {health_csv_path}")

    # Ingesta dinamica del dataset de logistica preparado
    if _table_needs_refresh(
        cursor,
        "inventario",
        stock_csv_path,
        ("Fecha", "ID_Insumo", "Stock_Actual", "Demanda_Fut_7d", "Demanda_Fut_14d"),
    ):
        if stock_csv_path and os.path.exists(stock_csv_path):
            print(f"Sembrando tabla 'inventario' desde {stock_csv_path}...")
            pd.read_csv(stock_csv_path).to_sql("inventario", conn, if_exists="replace", index=False)
        else:
            print(f"ADVERTENCIA: dataset de inventario no encontrado en {stock_csv_path}")

    # Indice unico sobre Patient_ID para habilitar upsert (ON CONFLICT) en pacientes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pacientes';")
    if cursor.fetchone() is not None:
        cols = [r[1] for r in cursor.execute("PRAGMA table_info(pacientes);").fetchall()]
        if "Patient_ID" in cols:
            try:
                cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_pacientes_id ON pacientes(Patient_ID);")
            except Exception as e:
                print(f"No se pudo crear indice unico en pacientes: {e}")

    conn.commit()
    conn.close()


def fetch_all_pacientes(db_path: str) -> pd.DataFrame:
    """Retorna todos los pacientes de la base de datos."""
    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM pacientes;", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def fetch_all_inventario(db_path: str) -> pd.DataFrame:
    """Retorna el historial de inventario ordenado por fecha."""
    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM inventario;", conn)
        if not df.empty and "Fecha" in df.columns:
            df["Fecha"] = pd.to_datetime(df["Fecha"])
            df = df.sort_values(["ID_Insumo", "Fecha"]) if "ID_Insumo" in df.columns else df.sort_values("Fecha")
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def _dynamic_upsert(db_path: str, table: str, data: dict, pk: str | None = None) -> bool:
    """Insert (o insert-or-replace si hay PK) generico basado en un dict."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    columns = list(data.keys())
    placeholders = ", ".join(["?"] * len(columns))
    col_str = ", ".join(columns)
    try:
        if pk and pk in columns:
            update_str = ", ".join([f"{c}=excluded.{c}" for c in columns if c != pk])
            query = (
                f"INSERT INTO {table} ({col_str}) VALUES ({placeholders}) "
                f"ON CONFLICT({pk}) DO UPDATE SET {update_str};"
            )
        else:
            query = f"INSERT INTO {table} ({col_str}) VALUES ({placeholders});"
        cursor.execute(query, [data[c] for c in columns])
        conn.commit()
        ok = True
    except Exception as e:
        # Degradar a INSERT simple si el upsert por PK no es soportado
        try:
            cursor.execute(
                f"INSERT INTO {table} ({col_str}) VALUES ({placeholders});",
                [data[c] for c in columns],
            )
            conn.commit()
            ok = True
        except Exception as e2:
            print(f"Error insertando en {table}: {e} / {e2}")
            ok = False
    finally:
        conn.close()
    return ok


def insert_paciente(db_path: str, patient_data: dict) -> bool:
    """Inserta o actualiza un paciente (clave Patient_ID)."""
    return _dynamic_upsert(db_path, "pacientes", patient_data, pk="Patient_ID")


def insert_inventario(db_path: str, stock_data: dict) -> bool:
    """Inserta un registro de inventario."""
    return _dynamic_upsert(db_path, "inventario", stock_data)


def save_prediction_riesgo(db_path: str, patient_id: int, prioridad: str, probas: tuple):
    """Guarda una prediccion de prioridad clinica (Bajo/Medio/Alto)."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO predicciones_riesgo
                (Patient_ID, Prioridad_Pred, Proba_Bajo, Proba_Medio, Proba_Alto)
            VALUES (?, ?, ?, ?, ?);
            """,
            (patient_id, prioridad, float(probas[0]), float(probas[1]), float(probas[2])),
        )
        conn.commit()
    except Exception as e:
        print(f"Error guardando prediccion de riesgo: {e}")
    finally:
        conn.close()


def save_prediction_stock(db_path: str, id_insumo: str, fecha: str, horizonte: str,
                          stock_proyectado: float, alerta: str):
    """Guarda una prediccion de stock proyectado."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO predicciones_stock
                (ID_Insumo, Fecha, Horizonte, Stock_Proyectado, Alerta)
            VALUES (?, ?, ?, ?, ?);
            """,
            (id_insumo, fecha, horizonte, float(stock_proyectado), alerta),
        )
        conn.commit()
    except Exception as e:
        print(f"Error guardando prediccion de stock: {e}")
    finally:
        conn.close()
