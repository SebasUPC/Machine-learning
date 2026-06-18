# ALDIMI Core AI — Machine Learning

Dashboard interactivo para priorización clínica, inventario predictivo y análisis estadístico del proyecto ALDIMI 2.0.

## Requisitos

- Python 3.10+
- Datos procesados en `data/processed/` (generados por los notebooks del repositorio)

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecutar el dashboard

```bash
streamlit run streamlit_app.py
```

## Estructura principal

| Ruta | Descripción |
|---|---|
| `streamlit_app.py` | Dashboard ALDIMI Core AI |
| `data/processed/` | CSV consolidados (salud, inventario) |
| `src/*.ipynb` | Pipeline de datos, modelado y evaluación |

## Módulos del dashboard

- **Operación diaria:** Resumen ejecutivo, Inventario predictivo, Priorización clínica
- **Análisis de datos:** Menú estadístico
- **Estrategia y gobierno:** MLOps, Impacto ODS y ética, Ecosistema Core AI
