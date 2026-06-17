# SECCIONES III, IV y V — Informe PC2 Machine Learning
## Responsable: Fabian Marcelo Rojas Cuadros (u202218498)
## Tema: Implementación, Dashboard, MLOps y Ecosistema ALDIMI Core AI

> **Instrucciones de uso:** Copiar cada sección numerada al informe Word, después de la Sección 6 (Evaluation). Insertar capturas del dashboard donde se indique [CAPTURA]. Ajustar numeración si el documento final usa otro esquema.

---

# III. Implementación y Despliegue (MLOps)

## III.1. Introducción a la capa de implementación

Tras completar las fases de comprensión del negocio, exploración de datos, preparación, modelado y evaluación, el proyecto requiere una **capa de implementación** que traduzca los resultados analíticos en una herramienta operativa utilizable por ALDIMI. Esta capa no constituye un producto médico ni un sistema de diagnóstico clínico; se define como un **sistema de apoyo a la decisión** orientado a la priorización preventiva de pacientes y a la anticipación logística de insumos críticos.

La implementación se materializa en dos artefactos complementarios:

1. **Pipeline MLOps reproducible**: flujo automatizado desde la ingesta de datos hasta la generación de predicciones, documentado en notebooks versionados y artefactos en `data/processed/`.
2. **Dashboard ALDIMI Core AI**: interfaz web desarrollada en Streamlit que consume los datos procesados, entrena modelos en tiempo de ejecución (con caché) y presenta KPIs, alertas, simuladores y análisis estadístico interactivo.

Este diseño responde al **Hito 4** del plan SCRUM del proyecto (Semanas 13–15): despliegue del dashboard final, validación con datos experimentales y reporte de impacto social.

---

## III.2. Arquitectura del pipeline de datos (MLOps)

### III.2.1. Visión general del flujo

El pipeline sigue una arquitectura por capas que separa responsabilidades y facilita el mantenimiento cuando ALDIMI escale de 50 a 100 familias:

```
[Fuentes externas - Kaggle]
         |
         v
[Capa 1: Ingesta]           --> data/raw/
         |
         v
[Capa 2: Integración]       --> data/merged/
         |
         v
[Capa 3: Preprocesamiento]    --> data/processed/
         |
         +------------------+
         |                  |
         v                  v
[Capa 4: Modelado]    [Capa 5: Dashboard]
  (notebooks ML)         (streamlit_app.py)
         |                  |
         v                  v
[Metricas y artefactos] [KPIs, alertas, simuladores]
```

Cada capa produce artefactos verificables (archivos CSV, notebooks ejecutados, métricas registradas) que permiten trazabilidad — principio fundamental de MLOps académico y profesional.

### III.2.2. Detalle por capa

**Capa 1 — Ingesta de datos (`01_descarga_datos.ipynb`)**

| Elemento | Detalle |
|----------|---------|
| Fuente clínica | Cancer Risk Factors Dataset (Kaggle) |
| Fuente logística | High-Dimensional Supply Chain Inventory Dataset (Kaggle) |
| Método | API oficial `kagglehub.dataset_download()` |
| Salida | `health_raw.csv`, `stock_raw.csv` en `data/raw/` |
| Responsable pipeline | Equipo ML — notebook 01 |

**Capa 2 — Integración (`02_merge_datasets.ipynb`)**

| Elemento | Detalle |
|----------|---------|
| Operación | Fusión de factores clínicos con variables socioeconómicas por condado |
| Registros resultantes | Más de 3,000 registros integrados |
| Salida principal | `Dataset_ALDIMI_Merged.csv` en `data/merged/` |
| Valor de negocio | Perfil 360° del paciente (clínico + contexto social) |

**Capa 3 — Preprocesamiento (`Preprocesamiento.ipynb`)**

| Técnica | Aplicación | Justificación |
|---------|------------|-----------------|
| Estandarización Z-score | Variables numéricas | Evita que escalas distintas distorsionen el modelo |
| One-Hot Encoding | Variables categóricas nominales | Previene jerarquías artificiales |
| Label Encoding | Variable objetivo multiclase | Permite entrenamiento supervisado |
| SMOTE | Balanceo de clases de riesgo | Mitiga sesgo hacia clases mayoritarias |
| Feature engineering | `Risk_Lifestyle_Score`, `Diet_Risk_Index` | Sintetiza factores dispersos en indicadores accionables |
| Split train/test | 80/20 estratificado | Evaluación honesta del desempeño |

Salidas en `data/processed/`:
- `Dataset_ALDIMI_Merged_Clean.csv` — dataset clínico limpio
- `stock_structured.csv` — inventario con series temporales
- `health_daily.csv` — agregación diaria de salud
- `train.csv` / `test.csv` — particiones para experimentación

**Capa 4 — Modelado (`Tarea_4.ipynb`, `04_predicciones.ipynb`)**

| Problema | Algoritmos | Variable objetivo | Métrica crítica |
|----------|------------|-------------------|-----------------|
| Clasificación de prioridad | Random Forest, XGBoost | `Risk_Level` (Bajo/Medio/Alto) | F1-Score clase Alto Riesgo |
| Regresión de inventario | Random Forest, XGBoost | `Stock_Actual` | MAE (interpretable en unidades) |

Modelo seleccionado: **XGBoost** (Accuracy 96.16%, F1-Score 96.08%), por superioridad en la métrica crítica y eficiencia computacional.

Control de calidad MLOps: se detectó y corrigió **data leakage** excluyendo `Overall_Risk_Score` del entrenamiento, variable numérica equivalente al target categórico.

**Capa 5 — Despliegue (`streamlit_app.py`)**

| Función | Implementación técnica |
|---------|------------------------|
| Carga de datos | `@st.cache_data` — lectura de CSV procesados |
| Entrenamiento | `@st.cache_resource` — pipelines sklearn/XGBoost en memoria |
| Predicción en vivo | Simuladores clínico y logístico con `predict()` / `predict_proba()` |
| Visualización | Plotly Express / Graph Objects |
| Navegación | Sidebar agrupado en 3 áreas (Operación, Análisis, Estrategia) |

---

## III.3. Especificación técnica de la solución de software

### III.3.1. Stack tecnológico

| Componente | Tecnología | Versión mínima | Rol |
|------------|------------|----------------|-----|
| Lenguaje | Python | 3.10+ | Desarrollo integral |
| Manipulación de datos | Pandas, NumPy | 2.0 / 1.24 | ETL en memoria |
| Machine Learning | scikit-learn | 1.3+ | Pipelines, métricas, preprocesamiento |
| Modelos avanzados | XGBoost | 2.0+ | Clasificación y regresión |
| Interfaz de usuario | Streamlit | 1.35+ | Dashboard web |
| Visualización | Plotly | 5.20+ | Gráficos interactivos |
| Control de versiones | Git / GitHub | — | Trazabilidad del código |
| Entorno | venv / conda | — | Aislamiento de dependencias |

Archivo de dependencias: `requirements.txt` en la raíz del repositorio.

### III.3.2. Estructura del repositorio

```
Trabajo_Final/
├── streamlit_app.py          # Dashboard ALDIMI Core AI
├── requirements.txt          # Dependencias Python
├── README.md                 # Instrucciones de ejecución
├── data/
│   ├── raw/                  # Datos crudos de Kaggle
│   ├── merged/               # Dataset fusionado
│   └── processed/            # Datos listos para modelado y dashboard
├── src/
│   ├── 01_descarga_datos.ipynb
│   ├── 02_merge_datasets.ipynb
│   ├── Preprocesamiento.ipynb
│   ├── EDA.ipynb
│   ├── Tarea_4.ipynb
│   └── 04_predicciones.ipynb
└── docs/
    └── Seccion_III_IV_V_Fabian.md
```

### III.3.3. Contrato de datos entre ML y Dashboard

El dashboard consume exclusivamente archivos de `data/processed/`, estableciendo un **contrato de interfaz** entre el equipo de modelado y la capa de presentación:

| Archivo | Columnas clave | Uso en dashboard |
|---------|----------------|------------------|
| `Dataset_ALDIMI_Merged_Clean.csv` | Patient_ID, Cancer_Type, Age, BMI, Risk_Level, variables clínicas y de condado | KPIs clínicos, clasificación, menú estadístico |
| `stock_structured.csv` | ID_Insumo, Fecha, Stock_Actual, Consumo_Diario, Lead_Time, Ocupacion_Albergue | Alertas, proyecciones 7/14 días, regresión |
| `health_daily.csv` | Fecha, agregaciones de salud | Series temporales (extensión futura) |

Variables derivadas calculadas en el dashboard (no en CSV):
- `Risk_Lifestyle_Score` = promedio(Smoking, Alcohol_Use, Obesity, Air_Pollution, Occupational_Hazards)
- `Diet_Risk_Index` = promedio(Diet_Red_Meat, Diet_Salted_Processed, 10 − Fruit_Veg_Intake)
- `Consumo_7d`, `Consumo_14d` = medias móviles por insumo
- `Cobertura_Dias` = Stock_Actual / Consumo_7d
- `Alerta` = Crítico | Preventivo | Normal según cobertura vs lead time

---

## III.4. Dashboard de visualización — Especificación funcional

### III.4.1. Principios de diseño UX

Para evitar que la navegación se perciba densa ante directivos no técnicos, el dashboard aplica tres principios:

1. **Jerarquía por áreas**: el menú lateral agrupa 7 secciones en 3 áreas temáticas, reduciendo la carga cognitiva.
2. **Lectura en 30 segundos**: la vista "Resumen ejecutivo" concentra KPIs, semáforo operativo y distribución de riesgo sin requerir interacción previa.
3. **Profundidad bajo demanda**: simuladores, matrices de confusión y estadísticas avanzadas se ubican en expanders o secciones secundarias.

[CAPTURA 1: Vista general del sidebar con las 3 áreas]

### III.4.2. Funcionalidades por sección

#### A. Resumen ejecutivo

| Funcionalidad | Descripción | Usuario objetivo |
|---------------|-------------|------------------|
| KPIs globales | Pacientes analizados, alto riesgo, alertas de stock, cobertura promedio | Dirección |
| Tarjetas de características | Resumen visual de las 3 capacidades principales | Directivos no técnicos |
| Guía rápida | Instrucciones en 3 pasos (expandible) | Nuevos usuarios |
| Semáforo operativo | Barras horizontales de cobertura por insumo con color por alerta | Logística |
| Distribución de riesgo | Gráfico de dona Bajo/Medio/Alto | Asistencia social |
| Comparación de modelos | Tablas RF vs XGBoost (F1, MAE, RMSE, R²) | Equipo técnico / dirección |

[CAPTURA 2: Resumen ejecutivo con KPIs y semáforo]

#### B. Inventario predictivo

| Funcionalidad | Descripción | Interactividad |
|---------------|-------------|----------------|
| Selector de insumo | Filtra serie temporal por producto | Dropdown |
| Horizonte 7/14 días | Alterna proyección de stock | Radio buttons |
| Gráfico temporal | Stock actual vs proyección vs umbral lead time | Plotly interactivo |
| Lista priorizada | Tabla ordenada por cobertura con alertas | Multiselect de filtros |
| Simulador de compras | Ajuste de consumo, lead time, ocupación → stock estimado y alerta | Sliders y number inputs |
| Descarga CSV | Exportar lista de reposición | Botón download |
| Recomendación automática | Mensaje Crítico/Preventivo/Normal con acción sugerida | Condicional |

[CAPTURA 3: Inventario predictivo con simulador]

#### C. Priorización clínica

| Funcionalidad | Descripción | Interactividad |
|---------------|-------------|----------------|
| Filtro por nivel de riesgo | Bajo, Medio, Alto | Multiselect |
| Filtro por tipo de cáncer | Subconjunto de diagnósticos | Multiselect |
| Filtro historial familiar | Solo pacientes con antecedentes | Toggle |
| Scatter de riesgo | Risk_Lifestyle_Score vs Diet_Risk_Index coloreado por prioridad | Hover con Patient_ID |
| Tabla de seguimiento | Top 30 pacientes sugeridos | Ordenado por score |
| Matriz de confusión | Heatmap RF/XGBoost en validación | Selectbox de modelo |
| Simulador de paciente | Sliders clínicos → probabilidad por clase | 12+ controles |
| Disclaimer ético | "No reemplaza evaluación médica" | Texto visible |

[CAPTURA 4: Simulador clínico con probabilidades]

#### D. Menú estadístico

Sección dedicada que replica y extiende el EDA del informe de forma interactiva:

| Análisis | Dataset | Herramientas |
|----------|---------|--------------|
| Distribuciones | Clínico / Inventario | Histogramas con boxplot marginal |
| Correlaciones | Clínico / Inventario | Heatmap Pearson |
| Dispersión | Clínico / Inventario | Scatter plots configurables |
| Outliers (Z-score) | Clínico / Inventario | Umbral \|Z\| ajustable + tabla + boxplot |
| Resumen descriptivo | Clínico / Inventario | describe() + descarga CSV |
| Análisis por categoría | Clínico | Historial familiar vs riesgo; top tipos de cáncer |
| Alertas por insumo | Inventario | Cobertura por producto |

[CAPTURA 5: Menú estadístico — correlaciones o outliers]

#### E. MLOps (vista estratégica en dashboard)

Presenta la arquitectura, stack, instrucciones de despliegue y métricas de monitoreo — espejo documental de esta Sección III para usuarios que acceden solo al dashboard.

[CAPTURA 6: Vista MLOps del dashboard]

---

## III.5. Proceso de despliegue

### III.5.1. Despliegue académico (actual)

```bash
# 1. Clonar repositorio
git clone https://github.com/SebasUPC/Machine-learning.git
cd Machine-learning

# 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar dashboard
streamlit run streamlit_app.py
```

URL local por defecto: `http://localhost:8501`

### III.5.2. Escenarios de despliegue

| Escenario | Infraestructura | Caso de uso | Consideraciones |
|-----------|-----------------|-------------|-----------------|
| Demo académica | Streamlit Community Cloud | Evaluación PC2, presentación a ALDIMI | Gratuito; datos anonimizados |
| Piloto institucional | PC local en sede ALDIMI | Prueba con equipo operativo | Sin exposición a internet |
| Producción ALDIMI 2.0 | Docker + servidor + BD | 100 familias, datos reales | Autenticación, anonimización, MLflow |

### III.5.3. Ciclo de vida MLOps (reentrenamiento)

1. **Monitoreo**: registrar F1 alto riesgo, falsos negativos, MAE y % alertas críticas semanalmente.
2. **Detección de deriva**: comparar distribución de variables nuevas vs entrenamiento.
3. **Reentrenamiento**: ejecutar notebooks de preprocesamiento y modelado con datos actualizados.
4. **Validación**: comparar métricas nueva versión vs versión anterior; aprobar solo si mejora o mantiene F1 alto riesgo.
5. **Despliegue**: actualizar CSV en `data/processed/` y reiniciar dashboard (futuro: swap de modelo en MLflow).

---

## III.6. Métricas de monitoreo en producción

| Dominio | Métrica | Umbral sugerido | Acción si se incumple |
|---------|---------|-----------------|----------------------|
| Clínico | F1 alto riesgo | ≥ 0.90 | Revisar features y balanceo |
| Clínico | Falsos negativos (Alto→Bajo) | ≤ 10 casos en test | Auditoría + revisión humana reforzada |
| Inventario | MAE | Minimizar | Revisar variables de consumo |
| Inventario | R² | ≥ 0.70 | Evaluar más features temporales |
| Operativo | % insumos en alerta crítica | Monitorear tendencia | Escalar a logística |
| Operativo | Cobertura promedio (días) | ≥ lead time medio | Planificar reposición |

---

## III.7. Conclusión de la Sección III

La implementación MLOps de ALDIMI Core AI cierra el ciclo CRISP-DM con un despliegue funcional que conecta más de 3,000 registros procesados, modelos XGBoost validados y una interfaz Streamlit orientada a usuarios de negocio. La arquitectura por capas, el contrato de datos en `data/processed/` y el dashboard interactivo constituyen la base técnica sobre la cual ALDIMI 2.0 podrá evolucionar hacia un sistema de gestión inteligente escalable.

---

# IV. Análisis de Impacto ODS y Ética de Datos

## IV.1. Marco de impacto social

El proyecto ALDIMI Core AI se enmarca en la misión institucional de la Asociación de Voluntariado de Infancia y Familia: garantizar acompañamiento integral y gratuito a niños y adolescentes oncológicos en situación de pobreza. La analítica predictiva no sustituye el factor humano; lo **potencia** liberando al equipo de tareas repetitivas de registro y priorización manual.

La evaluación de impacto se estructura en tres dimensiones:
- Contribución a Objetivos de Desarrollo Sostenible (ODS)
- Cuantificación operativa basada en evidencia
- Gobernanza ética de datos y algoritmos

---

## IV.2. Contribución a los Objetivos de Desarrollo Sostenible

### IV.2.1. ODS 3 — Salud y bienestar

**Meta vinculada:** Reducir la mortalidad prematura por enfermedades no transmisibles y fortalecer la prevención.

**Contribución del proyecto:**
- El modelo de clasificación identifica pacientes en niveles Bajo, Medio y Alto de prioridad de atención, permitiendo que el equipo médico-social destine recursos limitados a quienes presentan mayor vulnerabilidad.
- Variables como `Risk_Lifestyle_Score` (predictor #1 en ambos modelos) y `Diet_Risk_Index` (Top 5) traducen factores de riesgo en indicadores comprensibles para intervenciones de nutrición y estilo de vida — áreas centrales en el albergue ALDIMI.
- El simulador clínico del dashboard muestra probabilidades por clase, facilitando conversaciones preventivas con familias sin emitir diagnósticos.

**Indicador operativo:** Proporción de pacientes clasificados como Alto Riesgo identificados tempranamente para seguimiento prioritario (visible en KPI del dashboard).

### IV.2.2. ODS 10 — Reducción de las desigualdades

**Meta vinculada:** Empoderar y promover la inclusión social de todos, independientemente de origen económico.

**Contribución del proyecto:**
- La integración de variables socioeconómicas a nivel de condado (`county_STATE`, `county_CTYNAME`, `county_POPESTIMATE2015`) enriquece el perfil del paciente más allá de lo clínico.
- El EDA confirmó que factores del entorno (BMI, contexto de condado) correlacionan con el nivel de riesgo, validando que la procedencia de provincias — realidad mayoritaria en ALDIMI — debe considerarse en la priorización.
- La clasificación basada en evidencia reduce la dependencia de criterios subjetivos que podrían favorecer inconscientemente a ciertos perfiles.

**Indicador operativo:** Inclusión de variables socioeconómicas en el 100% de los registros del dataset integrado.

### IV.2.3. ODS 2 — Hambre cero (dimensión logística)

**Meta vinculada:** Poner fin al hambre, lograr la seguridad alimentaria y mejorar la nutrición.

**Contribución del proyecto:**
- ALDIMI proporciona alimentación especializada durante la estadía de las familias. Un quiebre de stock en insumos alimentarios o clínicos compromete directamente la nutrición del paciente.
- Las proyecciones de inventario a 7 y 14 días, combinadas con alertas Crítico/Preventivo/Normal, permiten planificar reposiciones antes de alcanzar niveles que afecten la continuidad alimentaria.
- La reducción de desperdicio por vencimiento (mediante mejor planificación) contribuye indirectamente a la sostenibilidad de los recursos de la asociación.

**Indicador operativo:** Porcentaje de insumos en alerta crítica (monitoreado en dashboard) y días de cobertura promedio.

---

## IV.3. Cuantificación del impacto operativo

### IV.3.1. Evidencia de la literatura

Chirinos Gonzales y Vereau Jacobo (2025), en su estudio sobre optimización de inventarios farmacéuticos con Machine Learning, reportan mejoras significativas al emplear modelos de ensamble como Random Forest y XGBoost:

| Indicador | Mejora reportada | Relevancia para ALDIMI |
|-----------|------------------|------------------------|
| Precisión predictiva | +30.5% vs métodos tradicionales | Estimación de demanda de medicamentos e insumos |
| Reducción de rupturas de stock | −28.5% | Continuidad de tratamiento oncológico |
| Optimización de niveles de inventario | −25.7% de ineficiencia | Menor desperdicio de recursos donados |

Estas cifras provienen de un contexto farmacéutico; su aplicación a ALDIMI es **proyectada** como estimación de beneficio potencial, no como medición directa en la operación actual (fase experimental con datos públicos).

### IV.3.2. Proyección cuantitativa para ALDIMI 2.0

Bajo el escenario de expansión de 50 a 100 familias:

| Área | Situación actual (manual) | Proyección con ALDIMI Core AI | Estimación |
|------|---------------------------|-------------------------------|------------|
| Detección de quiebres de stock | Reactiva (cuando ya falta) | Preventiva (7–14 días antes) | −28.5% rupturas* |
| Tiempo de análisis de prioridad | Registro manual por paciente | Automatizado con revisión humana | ~60% reducción de tiempo administrativo** |
| Falsos negativos clínicos | No medido | 8 casos en test (XGBoost) | Monitoreo continuo |
| Pacientes alto riesgo detectados | Dependiente de criterio individual | 22 casos correctamente clasificados en test | +1 vs Random Forest |

\* Basado en Chirinos & Vereau (2025). \*\* Estimación cualitativa del equipo; pendiente de medición en piloto.

### IV.3.3. Análisis costo-beneficio social

| Concepto | Costo / Inversión | Beneficio social |
|----------|-------------------|------------------|
| Infraestructura | Herramientas open source (costo ~0) | Acceso democrático a analítica |
| Capacitación | Tiempo del equipo académico | Transferencia de conocimiento a ALDIMI |
| Riesgo de error clínico | Falsos negativos posibles | Mitigado con revisión humana obligatoria |
| Continuidad de tratamiento | — | Menor probabilidad de interrupción por desabastecimiento |
| Equidad | — | Priorización basada en evidencia multivariable |

---

## IV.4. Ética de datos y gobernanza algorítmica

### IV.4.1. Identificación de sesgos

| Tipo de sesgo | Manifestación en el proyecto | Nivel de riesgo |
|---------------|------------------------------|-----------------|
| Sesgo poblacional | Dataset de cáncer de adultos aplicado a contexto pediátrico | **Alto** |
| Sesgo geográfico | Datos de condados de EE.UU., no de provincias peruanas | **Medio** |
| Sesgo de representación | Clase "Alto Riesgo" minoritaria (~ proporción desbalanceada) | **Medio** |
| Sesgo de confirmación | Riesgo de que el equipo confíe excesivamente en el modelo | **Medio** |
| Sesgo algorítmico | Variables proxy de condición socioeconómica podrían penalizar grupos vulnerables | **Medio-Alto** |

### IV.4.2. Medidas de mitigación implementadas

| Riesgo | Medida técnica | Medida organizacional |
|--------|----------------|----------------------|
| Data leakage | Exclusión de `Overall_Risk_Score` del entrenamiento | Auditoría documentada en Sección 6 del informe |
| Desbalance de clases | SMOTE + class_weight="balanced" + F1 macro | Monitoreo por clase en dashboard |
| Falsos negativos clínicos | Priorización de F1 alto riesgo como métrica de selección | Revisión humana obligatoria; disclaimer en simulador |
| Privacidad | Datasets públicos anonimizados; sin nombres reales | Política de mínimo dato necesario en producción |
| Sobreconfianza | Probabilidades visibles (no solo clase predicha) | Capacitación: "herramienta de apoyo, no diagnóstico" |
| Sesgo poblacional | Documentado como limitación | Validación futura con datos pediátricos reales de ALDIMI |

### IV.4.3. Principios éticos adoptados

1. **Autonomía:** El equipo humano de ALDIMI conserva la decisión final. El modelo sugiere; no prescribe.
2. **Beneficencia:** El objetivo es detectar más casos de riesgo y prevenir quiebres de stock, nunca optimizar métricas a costa de pacientes.
3. **No maleficencia:** Los disclaimers y la exclusión de variables con leakage protegen contra daños por predicciones engañosas.
4. **Justicia:** La inclusión de variables socioeconómicas busca equidad, no discriminación; requiere auditoría periódica de impacto por subgrupo.
5. **Transparencia:** Matrices de confusión, importancia de variables y arquitectura del pipeline son documentadas y visibles.

### IV.4.4. Marco legal y normativo (Perú)

En una fase de producción con datos reales de pacientes pediátricos, el sistema deberá alinearse con:
- **Ley N° 29733** — Ley de Protección de Datos Personales del Perú
- **Consentimiento informado** de las familias para uso analítico de datos clínicos
- **Anonimización** de identificadores antes de alimentar modelos
- **Política de retención** de datos acorde a la finalidad institucional de ALDIMI

En la fase académica actual, al utilizar datasets públicos anonimizados de Kaggle, estos requisitos operativos quedan como **recomendaciones de gobernanza** para la transición a ALDIMI 2.0.

---

## IV.5. Conclusión de la Sección IV

El proyecto demuestra contribución tangible a los ODS 2, 3 y 10, con estimaciones cuantitativas respaldadas por literatura especializada. La gobernanza ética — especialmente la detección de leakage, el balanceo de clases y la revisión humana obligatoria — constituye un diferencial responsable frente a implementaciones que priorizan métricas sin considerar el contexto social y clínico de ALDIMI.

---

# V. Confluencia con el Ecosistema "ALDIMI Core AI"

## V.1. Definición del ecosistema

**ALDIMI Core AI** no es un modelo aislado ni un dashboard independiente: es el **núcleo analítico** de un ecosistema de gestión inteligente que articula datos, algoritmos, interfaces y usuarios en un flujo continuo de valor. El ecosistema se compone de cinco bloques:

```
+------------------------------------------------------------------+
|                    ECOSISTEMA ALDIMI CORE AI                      |
+------------------------------------------------------------------+
|  [1] Fuentes de datos    Kaggle, futuros registros operativos     |
|  [2] Base comun IA       Integracion curso Inteligencia Artificial  |
|  [3] Motor ML            Notebooks CRISP-DM + modelos RF/XGBoost  |
|  [4] Capa de negocio     Dashboard Streamlit (este entregable)    |
|  [5] Usuarios finales    Direccion, logistica, equipo clinico     |
+------------------------------------------------------------------+
|  Retroalimentacion: datos operativos --> reentrenamiento --> KPIs |
+------------------------------------------------------------------+
```

---

## V.2. Integración con la base común de datos del curso de IA

El curso de Inteligencia Artificial del equipo definió una **base común de integración** que estandariza cómo se almacenan y relacionan pacientes, insumos, ocupación del albergue y variables externas. El proyecto de Machine Learning consume y enriquece esta base:

| Capa del ecosistema | Aporte del curso IA | Aporte del curso ML | Punto de confluencia |
|---------------------|---------------------|---------------------|----------------------|
| Datos | Esquema unificado, diccionario de datos | Notebooks de merge y limpieza | `data/processed/*.csv` |
| Features | Variables operativas definidas | Feature engineering (scores de riesgo) | Columnas en dataset limpio |
| Modelos | — | RF + XGBoost entrenados y evaluados | Metricas en dashboard |
| Interfaz | — | Dashboard Streamlit | `streamlit_app.py` |
| Gobernanza | Politicas de acceso (futuro) | Control de leakage, metricas eticas | Secciones MLOps y Etica |

### V.2.1. Flujo de datos entre equipos

```
Equipo IA (base comun)
        |
        |  Esquema de integracion
        v
Equipo ML — Notebook 01 (descarga Kaggle)
        |
        v
Equipo ML — Notebook 02 (merge con variables socioeconomicas)
        |
        v
Equipo ML — Preprocesamiento (SMOTE, encoding, train/test)
        |
        +------> data/processed/  <------+
        |                                |
        v                                |
Equipo ML — Tarea_4 (modelado)          |
        |                                |
        v                                |
Fabian — streamlit_app.py  -------------+
        |
        v
Usuarios ALDIMI (directivos, logistica, asistencia social)
```

### V.2.2. Artefactos compartidos

| Artefacto | Productor | Consumidor | Formato |
|-----------|-----------|------------|---------|
| `health_raw.csv` | Notebook 01 | Notebook 02 | CSV |
| `stock_raw.csv` | Notebook 01 | Preprocesamiento | CSV |
| `Dataset_ALDIMI_Merged.csv` | Notebook 02 | Preprocesamiento, EDA | CSV |
| `Dataset_ALDIMI_Merged_Clean.csv` | Preprocesamiento | Tarea_4, Dashboard | CSV |
| `stock_structured.csv` | Preprocesamiento | Tarea_4, Dashboard | CSV |
| Modelos entrenados | Tarea_4 | Dashboard (re-entrena en cache) | Pipeline sklearn |
| Metricas de evaluacion | Tarea_4 | Dashboard + Informe | DataFrames |

---

## V.3. Rol del dashboard como capa de negocio

El dashboard implementado por Fabian Rojas cumple la función de **traductor** entre el lenguaje técnico del Machine Learning y el lenguaje operativo de ALDIMI:

| Resultado tecnico | Traduccion en dashboard | Decision de negocio |
|-------------------|-------------------------|---------------------|
| Probabilidad P(Alto Riesgo) = 0.72 | Metric "Resultado estimado: Alto" + barra de probabilidades | Priorizar seguimiento social |
| Cobertura_Dias = 5, Lead_Time = 10 | Alerta "Critico" en rojo | Iniciar reposicion urgente |
| F1 alto riesgo = 96% | Tabla comparativa de modelos | Confianza en automatizacion parcial |
| Correlacion Smoking-Riesgo = 0.05 | Heatmap en menu estadistico | Validar variables del EDA |
| 8 falsos negativos en test | Nota en seccion MLOps/Etica | Mantener revision medica |

### V.3.1. Usuarios del ecosistema y sus necesidades

| Usuario | Necesidad principal | Seccion del dashboard |
|---------|---------------------|----------------------|
| Direccion general | Vision consolidada, KPIs, impacto ODS | Resumen ejecutivo, Impacto ODS |
| Jefe de logistica | Alertas de stock, proyecciones, simulador de compras | Inventario predictivo |
| Equipo medico-social | Priorizacion de pacientes, seguimiento sugerido | Priorizacion clinica |
| Equipo tecnico / academico | Metricas, arquitectura, estadisticas | MLOps, Menu estadistico |
| Voluntariado (futuro) | Vista simplificada de alertas | Resumen ejecutivo (modo simplificado*) |

\* Modo simplificado: evolucion futura del dashboard.

---

## V.4. Lecciones aprendidas en la integración de sistemas

### V.4.1. Lecciones técnicas

1. **Estandarizar nombres de columnas desde el merge:** Diferencias de nomenclatura entre notebooks retrasaron la carga en el dashboard. Solución: diccionario de datos compartido en la base común IA.

2. **Separar experimentacion de produccion:** Los notebooks generan multiples versiones intermedias; solo `data/processed/` debe alimentar el dashboard. Esto evita que cambios experimentales rompan la interfaz.

3. **Detectar leakage antes del despliegue:** La variable `Overall_Risk_Score` producía metricas artificialmente perfectas (>99%). Su exclusion fue una leccion critica de honestidad intelectual que debe perpetuarse como checklist de MLOps.

4. **Cache inteligente en Streamlit:** `@st.cache_data` y `@st.cache_resource` reducen tiempos de carga de ~30s a ~2s en recargas, esencial para demos ante directivos.

5. **Feature engineering compartido:** Los indices `Risk_Lifestyle_Score` y `Diet_Risk_Index` se calculan tanto en notebooks como en el dashboard. Idealmente deben centralizarse en un modulo Python compartido (`src/features.py`) en futuras iteraciones.

### V.4.2. Lecciones organizativas

1. **Metodologia SCRUM fue efectiva:** Los hitos alineados con CRISP-DM permitieron entregas incrementales (datos → modelos → dashboard).

2. **Roles claros aceleran la integracion:** Cada integrante cubrio una fase; Fabian integro la salida de todos en la capa de despliegue.

3. **Comunicacion negocio-tecnica:** Las secciones de ODS y Etica del dashboard facilitan conversaciones con ALDIMI sin requerir conocimiento de ML.

4. **Documentacion simultanea al codigo:** El informe y el dashboard deben contar la misma historia; discrepancias generan desconfianza en usuarios.

---

## V.5. Roadmap de evolucion del ecosistema

| Fase | Plazo sugerido | Accion | Responsable |
|------|----------------|--------|-------------|
| Fase 1 (actual) | PC2 — Ciclo 7 | Dashboard Streamlit + datos publicos | Equipo ML |
| Fase 2 | Post PC2 | Piloto con datos anonimizados de ALDIMI | ALDIMI + equipo |
| Fase 3 | ALDIMI 2.0 | Base de datos relacional + API FastAPI | Equipo IA + ML |
| Fase 4 | Escalamiento | MLflow, CI/CD, autenticacion por roles | DevOps / ML |
| Fase 5 | Madurez | Notificaciones automaticas, reentrenamiento programado | Operaciones ALDIMI |

### V.5.1. Arquitectura objetivo (ALDIMI 2.0)

```
[Registros operativos ALDIMI]
         |
         v
[Base de datos PostgreSQL]  <--- Base comun IA
         |
         v
[API FastAPI — servicio de predicciones]
         |
    +----+----+
    |         |
    v         v
[Dashboard]  [Alertas email/WhatsApp]
 Streamlit
```

---

## V.6. Conclusión de la Sección V

La confluencia entre la base común de datos del curso de Inteligencia Artificial, el pipeline de Machine Learning del equipo y el dashboard ALDIMI Core AI demuestra que es posible construir un ecosistema analitico integrado incluso en un contexto academico. Las lecciones aprendidas — estandarizacion, separacion de capas, auditoria de leakage y diseno centrado en el usuario — constituyen activos reutilizables para la transicion de ALDIMI hacia su fase 2.0 con capacidad duplicada de atencion.

El repositorio publico del proyecto (https://github.com/SebasUPC/Machine-learning) centraliza codigo, datos procesados y documentacion, cumpliendo con el Anexo VI del informe y garantizando reproducibilidad.

---

# ANEXO: Checklist de capturas para el informe

| # | Seccion del dashboard | Que debe verse | Seccion del informe |
|---|----------------------|----------------|---------------------|
| 1 | Resumen ejecutivo | KPIs + semaforo + dona de riesgo | III.4.2-A |
| 2 | Inventario predictivo | Grafico temporal + simulador | III.4.2-B |
| 3 | Priorizacion clinica | Simulador con probabilidades | III.4.2-C |
| 4 | Menu estadistico | Heatmap o outliers Z-score | III.4.2-D |
| 5 | MLOps | Tabla de arquitectura + metricas | III.2 / III.5 |
| 6 | Impacto ODS y etica | Metricas ODS + matriz de riesgos | IV.2 / IV.4 |
| 7 | Ecosistema Core AI | Tabla de integracion entre modulos | V.2 |
| 8 | Sidebar | 3 areas de navegacion | III.4.1 |

---

# ANEXO: Tabla de trazabilidad requisito ↔ entregable

| Requisito de la PC2 (Fabian) | Entregable | Ubicacion |
|------------------------------|------------|-----------|
| Desarrollo seccion III MLOps | Seccion III completa + vista MLOps en dashboard | Informe + `streamlit_app.py` |
| Desarrollo seccion IV ODS y Etica | Seccion IV completa + vista Impacto ODS | Informe + `streamlit_app.py` |
| Desarrollo seccion V Ecosistema | Seccion V completa + vista Ecosistema | Informe + `streamlit_app.py` |
| Especificacion tecnica | III.3 (stack, estructura, contrato de datos) | Informe |
| Dashboard interactivo | 7 secciones, simuladores, descargas CSV | `streamlit_app.py` |
| Menu estadistico | Seccion D del dashboard | `streamlit_app.py` |
| Simplificar navegacion | 3 areas en sidebar + tarjetas + guia rapida | `streamlit_app.py` |
| Resaltar caracteristicas principales | Tarjetas de features + KPIs + semaforo | `streamlit_app.py` |
