# ALDIMI Core AI — Machine Learning

Dashboard interactivo para priorización clínica, inventario predictivo y análisis estadístico del proyecto ALDIMI 2.0.

## Requisitos

- Python 3.10 o superior
- Datos enriquecidos disponibles en `data/processed/`.

## Instalación

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows CMD

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si el comando `pip` da errores de launcher, usa siempre `python -m pip` como arriba. Así evitas depender de accesos directos rotos del sistema.

## Ejecutar el dashboard

```bash
python -m streamlit run streamlit_app.py
```

## Primer arranque

1. Clona el repositorio.
2. Crea y activa el entorno virtual.
3. Instala las dependencias con `python -m pip install -r requirements.txt`.
4. Lanza la app con `python -m streamlit run streamlit_app.py`.

La aplicación carga los datasets ya incluidos en `data/processed/`, así que no hace falta rehacer todo el pipeline para probar el dashboard.

## Estructura principal

| Ruta | Descripción |
|---|---|
| `streamlit_app.py` | Dashboard ALDIMI Core AI |
| `data/processed/Dataset_ALDIMI_GravedadPaciente_Enriquecido.csv` | Pacientes + features de riesgo |
| `data/processed/Dataset_ALDIMI_Logistica_Enriquecido.csv` | Inventario + ratio y reabastecimiento |
| `src/*.ipynb` | Pipeline de datos, modelado y evaluación |

## Notas para colaboradores

- No subas el entorno local de Python ni sus carpetas generadas automáticamente; el archivo `.gitignore` ya cubre los casos habituales.
- Si cambias los notebooks que regeneran los datos, asegúrate de volver a exportar los CSV enriquecidos en `data/processed/`.

## Módulos del dashboard

- **Operación diaria:** Resumen ejecutivo, Inventario predictivo, Priorización clínica
- **Análisis de datos:** Menú estadístico
- **Estrategia y gobierno:** MLOps, Impacto ODS y ética, Ecosistema Core AI
