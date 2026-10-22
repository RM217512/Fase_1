# Registro de Modificaciones del Dataset (MiBici)

Este documento detalla el historial de versiones del dataset y las transformaciones aplicadas en cada etapa, asegurando la reproducibilidad del experimento. Todas las versiones físicas están respaldadas como artefactos en MLflow.

## Versión 0: Datos Crudos (Raw)
* **Ubicación:** `data/raw/`
* **Origen:** Archivos CSV mensuales descargados del portal de MiBici (últimos 18 meses).
* **Estado:** Datos originales sin alteraciones. Contienen valores nulos, inconsistencias en fechas y outliers en la duración de los viajes.

## Versión 1: Dataset Limpio (Clean)
* **Ubicación:** `data/processed/mibici_limpio_v1.csv`
* **Script responsable:** `src/data/clean_data.py`
* **Modificaciones realizadas:**
  1. Concatenación de los 18 archivos mensuales en un solo DataFrame.
  2. Eliminación de registros con valores nulos en las columnas de fechas (`Inicio_del_viaje`, `Fin_del_viaje`).
  3. Cálculo de la columna `duracion_min` (en minutos).
  4. Eliminación de anomalías (outliers) en la duración del viaje utilizando el algoritmo *Isolation Forest* (tasa de contaminación: 5%).

## Versión 2: Feature Engineering (Modelado)
* **Ubicación:** `data/processed/mibici_modelado_v1.csv`
* **Script responsable:** `src/features/build_features.py`
* **Modificaciones realizadas:**
  1. Extracción de características temporales: `hora_del_viaje`, `dia_de_la_semana`, `es_fin_de_semana`.
  2. Derivación matemática: Creación de la columna `edad` calculada desde el año de nacimiento.
  3. Codificación categórica: Aplicación de *One-Hot Encoding* a la columna `Genero`.
  4. Eliminación de columnas de texto y fechas originales incompatibles con modelos matemáticos.

## Versión 3: Preprocesamiento y PCA (Final)
* **Ubicación:** `data/processed/mibici_pca_v1.csv`
* **Script responsable:** `src/features/preprocess.py`
* **Modificaciones realizadas:**
  1. Eliminación de columnas completamente vacías detectadas en el EDA (`id`, `latitude`, `longitude`, `dpcapacity`).
  2. Separación de variables identificadoras (`Viaje_Id`, `Usuario_Id`, etc.) para no afectar el modelo.
  3. Normalización matemática de las características utilizando `StandardScaler`.
  4. Reducción de dimensionalidad aplicando PCA (*Principal Component Analysis*) para conservar el 95% de la varianza original.