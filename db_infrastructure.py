import sqlite3
import os
from pathlib import Path
import pandas as pd

def get_connection(db_path: str):
    """Establece una conexión con la base de datos SQLite."""
    conn = sqlite3.connect(db_path)
    # Habilitar soporte de llaves foráneas
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db(db_path: str, health_csv_path: str, stock_csv_path: str):
    """Crea las tablas de la base de datos e ingesta los datos iniciales si está vacía."""
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
        
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # 1. Tabla de pacientes
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pacientes (
        Patient_ID INTEGER PRIMARY KEY,
        Cancer_Type TEXT,
        Age INTEGER,
        Gender INTEGER,
        Smoking INTEGER,
        Alcohol_Use INTEGER,
        Obesity INTEGER,
        Family_History INTEGER,
        Diet_Red_Meat INTEGER,
        Diet_Salted_Processed INTEGER,
        Fruit_Veg_Intake INTEGER,
        Physical_Activity INTEGER,
        Air_Pollution INTEGER,
        Occupational_Hazards INTEGER,
        BRCA_Mutation INTEGER,
        H_Pylori_Infection INTEGER,
        Calcium_Intake INTEGER,
        Overall_Risk_Score REAL,
        BMI REAL,
        Physical_Activity_Level INTEGER,
        Risk_Level TEXT,
        county_STATE INTEGER,
        county_CTYNAME TEXT,
        county_POPESTIMATE2015 REAL,
        Habitos_Riesgo INTEGER,
        Riesgo_Clinico INTEGER,
        Factor_Protector INTEGER,
        Balance_Riesgo INTEGER,
        Edad_Rango TEXT
    );
    """)

    # 2. Tabla de inventario
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inventario (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Fecha TEXT,
        ID_Insumo TEXT,
        Stock_Actual REAL,
        Consumo_Diario REAL,
        Lead_Time INTEGER,
        Ocupacion_Albergue REAL,
        Pacientes_Alto_Riesgo INTEGER,
        Ocupacion_Total INTEGER,
        Punto_Reorden REAL,
        Ratio_Stock REAL,
        Necesita_Reabastecimiento INTEGER
    );
    """)

    # 3. Tabla de predicciones de riesgo (historial/registro)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predicciones_riesgo (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Patient_ID INTEGER,
        Risk_Level_Pred TEXT,
        Proba_Bajo REAL,
        Proba_Medio REAL,
        Proba_Alto REAL,
        Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (Patient_ID) REFERENCES pacientes(Patient_ID) ON DELETE CASCADE
    );
    """)

    # 4. Tabla de predicciones de stock (historial/registro)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predicciones_stock (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ID_Insumo TEXT,
        Fecha TEXT,
        Horizonte TEXT,
        Stock_Proyectado REAL,
        Alerta TEXT,
        Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()

    # Ingestar pacientes si está vacía
    cursor.execute("SELECT COUNT(*) FROM pacientes;")
    if cursor.fetchone()[0] == 0:
        if os.path.exists(health_csv_path):
            print(f"Sembrando tabla 'pacientes' desde {health_csv_path}...")
            df_health = pd.read_csv(health_csv_path)
            # Asegurar tipos correctos
            df_health.to_sql("pacientes", conn, if_exists="append", index=False)
        else:
            print(f"ADVERTENCIA: Archivo de salud no encontrado en {health_csv_path}")

    # Ingestar inventario si está vacía
    cursor.execute("SELECT COUNT(*) FROM inventario;")
    if cursor.fetchone()[0] == 0:
        if os.path.exists(stock_csv_path):
            print(f"Sembrando tabla 'inventario' desde {stock_csv_path}...")
            df_stock = pd.read_csv(stock_csv_path)
            df_stock.to_sql("inventario", conn, if_exists="append", index=False)
        else:
            print(f"ADVERTENCIA: Archivo de inventario no encontrado en {stock_csv_path}")

    conn.close()

def fetch_all_pacientes(db_path: str) -> pd.DataFrame:
    """Retorna todos los pacientes de la base de datos SQLite."""
    conn = get_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM pacientes;", conn)
    conn.close()
    return df

def fetch_all_inventario(db_path: str) -> pd.DataFrame:
    """Retorna todo el historial de inventario de la base de datos SQLite."""
    conn = get_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM inventario ORDER BY Fecha ASC;", conn)
    if not df.empty and 'Fecha' in df.columns:
        df['Fecha'] = pd.to_datetime(df['Fecha'])
    conn.close()
    return df

def insert_paciente(db_path: str, patient_data: dict) -> bool:
    """Inserta o actualiza un paciente en la base de datos."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    columns = list(patient_data.keys())
    placeholders = ", ".join(["?"] * len(columns))
    col_str = ", ".join(columns)
    
    # Manejar insert or replace
    update_str = ", ".join([f"{col} = excluded.{col}" for col in columns if col != 'Patient_ID'])
    query = f"""
    INSERT INTO pacientes ({col_str})
    VALUES ({placeholders})
    ON CONFLICT(Patient_ID) DO UPDATE SET {update_str};
    """
    
    try:
        cursor.execute(query, [patient_data[c] for c in columns])
        conn.commit()
        success = True
    except Exception as e:
        print(f"Error insertando paciente: {e}")
        success = False
    finally:
        conn.close()
    return success

def insert_inventario(db_path: str, stock_data: dict) -> bool:
    """Inserta un registro diario de inventario."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    columns = list(stock_data.keys())
    placeholders = ", ".join(["?"] * len(columns))
    col_str = ", ".join(columns)
    
    query = f"""
    INSERT INTO inventario ({col_str})
    VALUES ({placeholders});
    """
    
    try:
        cursor.execute(query, [stock_data[c] for c in columns])
        conn.commit()
        success = True
    except Exception as e:
        print(f"Error insertando inventario: {e}")
        success = False
    finally:
        conn.close()
    return success

def save_prediction_riesgo(db_path: str, patient_id: int, risk_level: str, probas: tuple):
    """Guarda una predicción de riesgo clínico en la base de datos."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO predicciones_riesgo (Patient_ID, Risk_Level_Pred, Proba_Bajo, Proba_Medio, Proba_Alto)
        VALUES (?, ?, ?, ?, ?);
        """, (patient_id, risk_level, probas[0], probas[1], probas[2]))
        conn.commit()
    except Exception as e:
        print(f"Error guardando predicción de riesgo: {e}")
    finally:
        conn.close()

def save_prediction_stock(db_path: str, id_insumo: str, fecha: str, horizonte: str, stock_proyectado: float, alerta: str):
    """Guarda una predicción de stock en la base de datos."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO predicciones_stock (ID_Insumo, Fecha, Horizonte, Stock_Proyectado, Alerta)
        VALUES (?, ?, ?, ?, ?);
        """, (id_insumo, fecha, horizonte, stock_proyectado, alerta))
        conn.commit()
    except Exception as e:
        print(f"Error guardando predicción de stock: {e}")
    finally:
        conn.close()
