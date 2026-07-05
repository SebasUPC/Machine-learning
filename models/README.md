# Carpeta de modelos (artefactos .joblib)

Esta carpeta almacena los modelos entrenados en **Google Colab** con los
notebooks del Hito 3. Descargue estos archivos desde Colab y coloquelos aqui
para que `streamlit_app.py` y el notebook `11_Evaluacion.ipynb` los utilicen.

## Artefactos esperados

| Archivo | Generado por | Contenido |
|---|---|---|
| `clf_random_forest.joblib` | `09_Modelado_Clasificacion.ipynb` | Pipeline RF (clasificacion) |
| `clf_xgboost.joblib` | `09` | Pipeline XGBoost (clasificacion) |
| `modelo_clasificacion.joblib` | `09` | Modelo seleccionado + label encoder + features |
| `metricas_clasificacion.csv` | `09` | Metricas comparativas RF vs XGBoost |
| `reg_demanda_t7.joblib` / `reg_demanda_t14.joblib` | `10_Modelado_Regresion.ipynb` | Mejor regresor de demanda por horizonte |
| `reg_stock_t7.joblib` / `reg_stock_t14.joblib` | `10` (nombre alternativo) | Equivalente a `reg_demanda_*` en versiones anteriores |
| `reg_t7_*.joblib` / `reg_t14_*.joblib` | `10` | Candidatos RF/XGB por horizonte |
| `meta_regresion.joblib` | `10` | Features y targets de regresion (demanda) |
| `metricas_regresion.csv` | `10` | Metricas comparativas (Tablas 3 y 4) |

> **Nota:** si esta carpeta esta vacia, el dashboard entrena automaticamente
> modelos ligeros de demostracion (fallback) para no bloquear la visualizacion.
>
> **Compatibilidad Colab:** los pipelines de clasificacion requieren la misma
> version de scikit-learn que en Colab (`scikit-learn==1.6.1` en `requirements.txt`).
> Si la carga falla, el dashboard muestra un aviso y usa fallback.
