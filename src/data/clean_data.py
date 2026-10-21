import glob
import pandas as pd
from sklearn.ensemble import IsolationForest
import mlflow

def clean_and_log_data():
    # 1. Configurar el experimento en MLflow
    mlflow.set_experiment("Proyecto_MiBici_Limpieza")
    
    with mlflow.start_run(run_name="Limpieza_V1"):
        print("Cargando y uniendo archivos CSV...")
        archivos_csv = glob.glob('data/raw/*.csv')
        
        # Ordenar los archivos por nombre para asegurar el orden cronológico
        archivos_csv.sort()
        
        # Tomar únicamente los últimos 18 archivos (últimos 18 meses)
        archivos_csv = archivos_csv[-18:]
        
        print(f"Se procesarán {len(archivos_csv)} meses recientes.")
        
        # Cargar datos con encoding latin1 para evitar el error de caracteres (ñ, acentos)
        df_list = [pd.read_csv(archivo, low_memory=False, encoding='latin1') for archivo in archivos_csv] 
        df = pd.concat(df_list, ignore_index=True)
        
        filas_originales = len(df)
        print(f"Total de registros iniciales: {filas_originales}")

        print("Eliminando valores nulos e inconsistentes...")
        # Imprimir las columnas para saber exactamente cómo vienen nombradas en tu dataset
        print("Columnas detectadas:", df.columns.tolist())
        
        # En lugar de borrar todo, solo borramos las filas donde falte la fecha de inicio o fin
        # (Ajusta los nombres de las columnas si el print de arriba muestra que se llaman diferente)
        if 'Inicio_del_viaje' in df.columns and 'Fin_del_viaje' in df.columns:
            df = df.dropna(subset=['Inicio_del_viaje', 'Fin_del_viaje'])
        
        # Asegurar formato de fechas y crear variable objetivo para el filtrado
        # NOTA: Revisa que tus columnas se llamen exactamente 'Inicio_del_viaje' y 'Fin_del_viaje'
        if 'Inicio_del_viaje' in df.columns and 'Fin_del_viaje' in df.columns:
            df['Inicio_del_viaje'] = pd.to_datetime(df['Inicio_del_viaje'], errors='coerce')
            df['Fin_del_viaje'] = pd.to_datetime(df['Fin_del_viaje'], errors='coerce')
            df = df.dropna(subset=['Inicio_del_viaje', 'Fin_del_viaje'])
            
            # Calcular duración en minutos
            df['duracion_min'] = (df['Fin_del_viaje'] - df['Inicio_del_viaje']).dt.total_seconds() / 60
            df = df[df['duracion_min'] > 0] # Eliminar viajes con tiempos negativos o en ceros
        
        print("Detectando outliers con Isolation Forest...")
        tasa_contaminacion = 0.05
        iso_forest = IsolationForest(contamination=tasa_contaminacion, random_state=42)
        
        if 'duracion_min' in df.columns:
            # Entrenar el modelo para detectar anomalías en la duración del viaje
            outlier_labels = iso_forest.fit_predict(df[['duracion_min']])
            df = df[outlier_labels == 1] # Conservar solo inliers (1) y descartar outliers (-1)
            
            # Registrar parámetros en MLflow
            mlflow.log_param("algoritmo_outliers", "IsolationForest")
            mlflow.log_param("contamination", tasa_contaminacion)

        filas_finales = len(df)
        print(f"Total de registros después de limpieza: {filas_finales}")
        
        # Registrar métricas de trazabilidad en MLflow
        mlflow.log_metric("filas_originales", filas_originales)
        mlflow.log_metric("filas_finales", filas_finales)
        mlflow.log_metric("filas_eliminadas", filas_originales - filas_finales)

        print("Guardando dataset procesado...")
        ruta_salida = "data/processed/mibici_limpio_v1.csv"
        df.to_csv(ruta_salida, index=False)
        
        # Guardar el dataset limpio como un artefacto versionado
        mlflow.log_artifact(ruta_salida, artifact_path="versiones_dataset")
        print("¡Limpieza y versionamiento en MLflow completados exitosamente!")

if __name__ == "__main__":
    clean_and_log_data()