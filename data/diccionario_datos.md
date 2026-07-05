# Diccionario de Datos — ALDIMI Core AI (Anexo)

Este anexo documenta todas las variables de ambos frentes: las columnas crudas
de Kaggle y las variables derivadas (feature engineering) generadas en la fase
de preparacion (`src/aldimi_common.py`).

> **Cohorte pediatrico-juvenil (salud):** el EDA (notebook 03) y el dataset preparado (`Dataset_ALDIMI_Salud_Preparado.csv`, notebook 05) conservan solo pacientes con **`Age < 25`**, alineados con la poblacion de ALDIMI.

---

## Frente 2 — Salud (Clasificacion) · `biased_leukemia_dataset.csv` (143.194 x 22)

### Variables crudas

| # | Variable | Tipo (Python) | Tipo de Variable | Descripcion |
|---|----------|---------------|------------------|-------------|
| 1 | Patient_ID | int64 | Identificador | Codigo unico del paciente. |
| 2 | Age | int64 | Cuantitativa Discreta | Edad del paciente en anios. |
| 3 | Gender | object | Cualitativa Nominal | Sexo (Male/Female). |
| 4 | Country | object | Cualitativa Nominal | Pais de origen (22 categorias). |
| 5 | WBC_Count | int64 | Cuantitativa Discreta | Recuento de globulos blancos (celulas/uL). |
| 6 | RBC_Count | float64 | Cuantitativa Continua | Recuento de globulos rojos (millones/uL). |
| 7 | Platelet_Count | int64 | Cuantitativa Discreta | Recuento de plaquetas (celulas/uL). |
| 8 | Hemoglobin_Level | float64 | Cuantitativa Continua | Nivel de hemoglobina (g/dL). |
| 9 | Bone_Marrow_Blasts | int64 | Cuantitativa Discreta | Porcentaje de blastos en medula osea. |
| 10 | Genetic_Mutation | object | Cualitativa Nominal (Dicotomica) | Presencia de mutacion genetica (Yes/No). |
| 11 | Family_History | object | Cualitativa Nominal (Dicotomica) | Antecedente familiar de cancer (Yes/No). |
| 12 | Smoking_Status | object | Cualitativa Nominal (Dicotomica) | Exposicion a tabaco (Yes/No). |
| 13 | Alcohol_Consumption | object | Cualitativa Nominal (Dicotomica) | Consumo de alcohol (Yes/No). |
| 14 | Radiation_Exposure | object | Cualitativa Nominal (Dicotomica) | Exposicion a radiacion (Yes/No). |
| 15 | Infection_History | object | Cualitativa Nominal (Dicotomica) | Antecedente de infecciones (Yes/No). |
| 16 | BMI | float64 | Cuantitativa Continua | Indice de masa corporal. |
| 17 | Chronic_Illness | object | Cualitativa Nominal (Dicotomica) | Enfermedad cronica preexistente (Yes/No). |
| 18 | Immune_Disorders | object | Cualitativa Nominal (Dicotomica) | Trastornos inmunologicos (Yes/No). |
| 19 | Ethnicity | object | Cualitativa Nominal | Grupo etnico (3 categorias). |
| 20 | Socioeconomic_Status | object | Cualitativa Ordinal | Nivel socioeconomico (Low/Medium/High). |
| 21 | Urban_Rural | object | Cualitativa Nominal (Dicotomica) | Zona de residencia (Urban/Rural). |
| 22 | Leukemia_Status | object | Cualitativa Nominal (Dicotomica) | Diagnostico de leucemia (Positive/Negative). |

### Variables derivadas (feature engineering)

| Variable | Tipo | Descripcion |
|----------|------|-------------|
| `*_flag` | int (0/1) | Version numerica de cada variable dicotomica Yes/No. |
| Blastos_Altos | int (0/1) | Blastos >= 20% (criterio de severidad). |
| Hemoglobina_Baja | int (0/1) | Hemoglobina < 12 g/dL. |
| Plaquetas_Bajas | int (0/1) | Plaquetas < 150.000. |
| WBC_Anormal | int (0/1) | Globulos blancos fuera de 4.000-11.000. |
| Severidad_Clinica | int (0-6) | Suma de marcadores clinicos anormales + mutacion + inmune. |
| Habitos_Riesgo | int (0-4) | Tabaco + alcohol + radiacion + infecciones. |
| Riesgo_Antecedentes | int (0-3) | Historial familiar + cronico + mutacion. |
| Vulnerabilidad_Social | int (0-2) | Nivel socioeconomico bajo + zona rural (ODS 10). |
| Leucemia_Positiva | int (0/1) | Diagnostico confirmado. |
| Edad_Rango | object | Nino/Adolescente/Adulto_Joven/Adulto/Adulto_Mayor. |
| Prioridad_Score | float | Valoracion integral del equipo (referencia interna; no disponible en inferencia). |
| Indice_Riesgo_Clinico | float | Indice compuesto de riesgo clinico (severidad, diagnostico, antecedentes, habitos, vulnerabilidad social). |
| Score_Triage | float | Puntuacion de evaluacion inicial en admision (protocolo ALDIMI). |
| **Prioridad_Atencion** | object (ordinal) | **VARIABLE OBJETIVO**: Bajo / Medio / Alto. |

---

## Frente 1 — Logistica (Regresion) · `supply_chain_dataset1.csv` (91.250 x 15)

### Variables crudas

| # | Variable | Tipo (Python) | Tipo de Variable | Descripcion |
|---|----------|---------------|------------------|-------------|
| 1 | Date | object | Fecha | Dia del registro (2024, 365 dias). |
| 2 | SKU_ID | object | Identificador | Codigo de producto/insumo (50 SKUs). |
| 3 | Warehouse_ID | object | Cualitativa Nominal | Almacen (5). |
| 4 | Supplier_ID | object | Cualitativa Nominal | Proveedor (10). |
| 5 | Region | object | Cualitativa Nominal | Region (West/North/South/East). |
| 6 | Units_Sold | int64 | Cuantitativa Discreta | Unidades consumidas/vendidas en el dia. |
| 7 | Inventory_Level | int64 | Cuantitativa Discreta | Nivel de inventario al cierre del dia. |
| 8 | Supplier_Lead_Time_Days | int64 | Cuantitativa Discreta | Tiempo de entrega del proveedor (dias). |
| 9 | Reorder_Point | int64 | Cuantitativa Discreta | Punto de reorden (umbral de pedido). |
| 10 | Order_Quantity | int64 | Cuantitativa Discreta | Cantidad pedida al proveedor. |
| 11 | Unit_Cost | float64 | Cuantitativa Continua | Costo unitario. |
| 12 | Unit_Price | float64 | Cuantitativa Continua | Precio de venta unitario. |
| 13 | Promotion_Flag | int64 (binario) | Cualitativa Nominal (Dicotomica) | Producto en promocion (1/0). |
| 14 | Stockout_Flag | int64 (binario) | Cualitativa Nominal (Dicotomica) | Desabastecimiento (1/0). En estos datos todos = 0. |
| 15 | Demand_Forecast | float64 | Cuantitativa Continua | Pronostico de demanda del proveedor. |

### Variables derivadas (contexto ALDIMI + series temporales)

| Variable | Tipo | Descripcion |
|----------|------|-------------|
| Fecha | datetime | `Date` normalizada. |
| ID_Insumo | object | `SKU_ID`. |
| Insumo | object | Nombre del insumo ALDIMI (mapeo determinista). |
| Categoria_Insumo | object | Medicamento Oncologico / Alimento Especializado / Suministro Clinico / Higiene y Aseo. |
| Stock_Actual | int | `Inventory_Level` consolidado por insumo/dia. |
| Consumo_Diario | int | `Units_Sold` consolidado. |
| Lead_Time | int | `Supplier_Lead_Time_Days`. |
| Punto_Reorden | int | `Reorder_Point`. |
| Demanda_Pronosticada | float | `Demand_Forecast`. |
| Consumo_7d / Consumo_14d | float | Promedio movil de consumo. |
| Consumo_Prev_7d / Consumo_Prev_14d | float | Suma movil del consumo de los ultimos 7 / 14 dias (predictor de la demanda futura). |
| Consumo_Std_7d | float | Volatilidad (desviacion estandar) del consumo en 7 dias. |
| Consumo_Lag_1 / Consumo_Lag_7 | float | Rezago del consumo de 1 y 7 dias. |
| Stock_Lag_1 | float | Rezago del stock de 1 dia. |
| Ratio_Stock | float | Stock_Actual / Punto_Reorden. |
| Cobertura_Dias | float | Stock_Actual / Consumo_7d. |
| Mes / Dia_Semana | int | Componentes temporales. |
| Ocupacion_Total | int | Familias en el albergue (rampa 50->100, contexto ALDIMI 2.0). |
| Ocupacion_Albergue | float | Ocupacion normalizada (0-1). |
| Pacientes_Alto_Riesgo | int | Estimacion clinica de pacientes de alto riesgo. |
| Alerta | object | Normal / Preventivo / Critico (segun Ratio_Stock). |
| Necesita_Reabastecimiento | int | 0 / 1 / 2. |
| **Demanda_Fut_7d** | float | **TARGET**: demanda (consumo) acumulada de los proximos 7 dias. El stock proyectado se deriva como `Stock_Actual - Demanda_Fut_7d`. |
| **Demanda_Fut_14d** | float | **TARGET**: demanda (consumo) acumulada de los proximos 14 dias. |
