import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

from db_infrastructure import init_db


def test_init_db_rebuilds_pacientes_when_target_column_missing(tmp_path):
    health_csv = tmp_path / "health.csv"
    stock_csv = tmp_path / "stock.csv"

    pd.DataFrame(
        [
            {
                "Patient_ID": 1,
                "Prioridad_Score": 10.0,
                "Prioridad_Atencion": "Bajo",
            }
        ]
    ).to_csv(health_csv, index=False)

    pd.DataFrame([{"Fecha": "2024-01-01", "ID_Insumo": "SKU_1", "Stock_Actual": 10}]).to_csv(stock_csv, index=False)

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE pacientes (Patient_ID INTEGER PRIMARY KEY, old_col TEXT)")
    conn.execute("INSERT INTO pacientes (Patient_ID, old_col) VALUES (1, 'x')")
    conn.commit()
    conn.close()

    init_db(str(db_path), str(health_csv), str(stock_csv))

    conn = sqlite3.connect(db_path)
    columns = [row[1] for row in conn.execute("PRAGMA table_info(pacientes)").fetchall()]
    conn.close()

    assert "Prioridad_Atencion" in columns
    assert "Prioridad_Score" in columns
