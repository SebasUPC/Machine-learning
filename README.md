# ALDIMI Core AI — Machine Learning

Dashboard interactivo para priorización clínica, inventario predictivo y análisis estadístico del proyecto ALDIMI 2.0.

## Requisitos

- Python 3.10+
- Datos enriquecidos en `data/processed/` (salida del notebook `06_Enriquecimiento...`)

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecutar el dashboard

```bash
python -m streamlit run streamlit_app.py
```

## Estructura principal

| Ruta | Descripción |
|---|---|
| `streamlit_app.py` | Dashboard ALDIMI Core AI |
| `data/processed/Dataset_ALDIMI_GravedadPaciente_Enriquecido.csv` | Pacientes + features de riesgo |
| `data/processed/Dataset_ALDIMI_Logistica_Enriquecido.csv` | Inventario + ratio y reabastecimiento |
| `src/*.ipynb` | Pipeline de datos, modelado y evaluación |

## Módulos del dashboard

- **Operación diaria:** Resumen ejecutivo, Inventario predictivo, Priorización clínica
- **Análisis de datos:** Menú estadístico
- **Estrategia y gobierno:** MLOps, Impacto ODS y ética, Ecosistema Core AI
