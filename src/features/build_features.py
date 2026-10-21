import pandas as pd
import mlflow

def transformar_datos():
    # Creamos un nuevo experimento para separar la limpieza de la transformación
    mlflow.set_experiment("Proyecto_MiBici_Transformacion")
    
    with mlflow.start_run(run_name="Feature_Engineering_V1"):
        print("Cargando dataset limpio...")
        # Leemos el archivo que generaste en el Punto 2
        df = pd.read_csv('data/processed/mibici_limpio_v1.csv', low_memory=False)
        
        # 1. Transformaciones Temporales
        print("Extrayendo características temporales...")
        df['Inicio_del_viaje'] = pd.to_datetime(df['Inicio_del_viaje'])
        df['hora_del_viaje'] = df['Inicio_del_viaje'].dt.hour
        df['dia_de_la_semana'] = df['Inicio_del_viaje'].dt.dayofweek # 0=Lunes, 6=Domingo
        df['es_fin_de_semana'] = df['dia_de_la_semana'].apply(lambda x: 1 if x >= 5 else 0)
        
        # 2. Derivación Matemática (Edad)
        print("Calculando edad de los usuarios...")
        # Usamos 2026 (tu año actual) para el cálculo exacto
        if 'Año_de_nacimiento' in df.columns:
            df['edad'] = 2026 - df['Año_de_nacimiento']
            
        # 3. Codificación de variables categóricas (One-Hot Encoding)
        # Esto convierte "Genero" en columnas de 1s y 0s (ej. Genero_M, Genero_F)
        print("Codificando variable Género...")
        if 'Genero' in df.columns:
            df = pd.get_dummies(df, columns=['Genero'], drop_first=True)
            
        # 4. Descarte de columnas incompatibles con ML
        # Eliminamos IDs de texto, nombres o fechas completas que el modelo no puede procesar
        columnas_a_eliminar = ['Inicio_del_viaje', 'Fin_del_viaje', 'name', 'Año_de_nacimiento']
        df = df.drop(columns=[col for col in columnas_a_eliminar if col in df.columns])
        
        print("Guardando dataset transformado para modelado...")
        ruta_salida = "data/processed/mibici_modelado_v1.csv"
        df.to_csv(ruta_salida, index=False)
        
        # Registrar metadatos en MLflow
        mlflow.log_param("transformaciones", "hora, dia_semana, edad, OHE_genero")
        mlflow.log_metric("total_features", len(df.columns))
        mlflow.log_artifact(ruta_salida, artifact_path="dataset_modelado")
        
        print(f"¡Transformación completada! El dataset ahora tiene {len(df.columns)} columnas numéricas.")

if __name__ == "__main__":
    transformar_datos()