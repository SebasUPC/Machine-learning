# ALDIMI Core AI — Ecosistema de Gestion Inteligente (ALDIMI 2.0)

Motor analitico predictivo para el Albergue Divina Misericordia (ALDIMI), una
organizacion sin fines de lucro que brinda atencion oncologica pediatrica
gratuita. El proyecto aplica la metodologia **CRISP-DM** (fases 1 a 6) sobre dos
frentes de trabajo:

- **Frente 1 — Logistica (Regresion):** prediccion de la **demanda (consumo) de insumos**
  a horizontes de **7 y 14 dias**; de ella se deriva el stock proyectado y las alertas de
  reposicion para sostener la expansion de 50 a 100 familias.
- **Frente 2 — Salud (Clasificacion):** priorizacion preventiva de pacientes **pediatrico-juveniles** (`Age < 25`) en niveles **Bajo / Medio / Alto**.

El proyecto se alinea con los **ODS 3 (Salud y Bienestar)** y **ODS 10
(Reduccion de las Desigualdades)**.

## Datasets (descargados con kagglehub)

| Frente | Dataset Kaggle | Archivo |
|---|---|---|
| Salud (clasificacion) | `ankushpanday1/leukemia-cancer-risk-prediction-dataset` | `biased_leukemia_dataset.csv` |
| Logistica (regresion) | `ziya07/high-dimensional-supply-chain-inventory-dataset` | `supply_chain_dataset1.csv` |

## Estructura del repositorio

```
finTF/
├── data/
│   ├── raw/            # CSV crudos de Kaggle
│   ├── interim/        # datos limpios intermedios
│   └── processed/      # datasets finales + aldimi.db
├── Hito1_Comprension_Negocio_Datos/
│   ├── 01_Business_Understanding.ipynb
│   ├── 02_Adquisicion_Datos.ipynb
│   ├── 03_EDA_Salud.ipynb
│   └── 04_EDA_Logistica.ipynb
├── Hito2_Preparacion_Baseline/
│   ├── 05_Preparacion_Salud.ipynb
│   ├── 06_Preparacion_Logistica.ipynb
│   ├── 07_Integracion_BD.ipynb
│   └── 08_Baselines.ipynb
├── Hito3_Modelado_Avanzado_Evaluacion/   # LISTO PARA GOOGLE COLAB
│   ├── 09_Modelado_Clasificacion.ipynb
│   ├── 10_Modelado_Regresion.ipynb
│   └── 11_Evaluacion.ipynb
├── models/             # artefactos .joblib (generados en Colab)
├── reports/figuras/    # exportacion opcional de figuras
├── src/aldimi_common.py    # rutas, mapeos y feature engineering compartido
├── db_infrastructure.py    # capa de base de datos (SQLite)
├── streamlit_app_Dash.py   # dashboard unico ALDIMI Core AI
└── requirements.txt
```

## Como ejecutar

1. Crear entorno e instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Ejecutar los notebooks en orden (Hito 1 -> Hito 2). El Hito 2 genera los
   datasets procesados en `data/processed/` y la base `aldimi.db`.
3. Entrenar los modelos avanzados en **Google Colab** con los notebooks del
   Hito 3 (generan los `.joblib` en `models/`).
4. Lanzar el dashboard:
   ```bash
   streamlit run streamlit_app_Dash.py
   ```

## Flujo CRISP-DM y entregables

| Fase CRISP-DM | Notebook(s) | Hito |
|---|---|---|
| 1. Business Understanding | `01` | Hito 1 |
| 2. Data Understanding + EDA | `02`, `03`, `04` | Hito 1 |
| 3. Data Preparation + Integracion | `05`, `06`, `07` | Hito 2 |
| 4. Modeling (baseline) | `08` | Hito 2 |
| 4. Modeling (avanzado) | `09`, `10` | Hito 3 (Colab) |
| 5. Evaluation | `11` | Hito 3 (Colab) |
| 6. Deployment | `streamlit_app_Dash.py` | Hito 4 |

## Persistencia de modelos

Se utiliza **joblib** (recomendado frente a pickle para modelos de
scikit-learn/XGBoost por su eficiencia con arrays NumPy). Los notebooks del
Hito 3 se ejecutan en Colab y depositan los artefactos en `models/`; el
dashboard los carga desde alli.

## Repositorio

https://github.com/SebasUPC/Machine-learning
