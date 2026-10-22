import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import mlflow
import os

def preprocesar_datos():
    mlflow.set_experiment("Proyecto_MiBici_Preprocesamiento")
    
    with mlflow.start_run(run_name="Normalizacion_y_PCA_V2"):
        print("Cargando dataset modelado...")
        df = pd.read_csv('data/processed/mibici_modelado_v1.csv', low_memory=False)

        # 1. Limpieza final de datos nulos detectados en el EDA
        columnas_vacias = ['id', 'latitude', 'longitude', 'dpcapacity']
        df = df.drop(columns=[col for col in columnas_vacias if col in df.columns])

        # 2. Separar características (X) de los identificadores
        # Los IDs no aportan valor matemático al modelo, así que los separamos temporalmente
        cols_identificadores = ['Viaje_Id', 'Usuario_Id', 'Origen_Id', 'Destino_Id']
        df_features = df.drop(columns=[col for col in cols_identificadores if col in df.columns])
        
        # Rellenar cualquier NaN residual con la mediana
        df_features = df_features.fillna(df_features.median())

        # 3. Normalización
        print("Aplicando Normalización (StandardScaler)...")
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_features)

        # 4. Reducción de Dimensionalidad
        print("Aplicando Reducción de Dimensionalidad (PCA)...")
        # Le pedimos al algoritmo que conserve el 95% de la varianza original de los datos
        pca = PCA(n_components=0.80, random_state=42)
        df_pca = pca.fit_transform(df_scaled)

        columnas_originales = df_features.shape[1]
        columnas_nuevas = df_pca.shape[1]
        print(f"Dimensiones reducidas de {columnas_originales} a {columnas_nuevas} componentes.")

        # Reconstruir el DataFrame final
        columnas_pca = [f"PC{i+1}" for i in range(columnas_nuevas)]
        df_final = pd.DataFrame(df_pca, columns=columnas_pca)

        # Reincorporar los IDs para no perder la trazabilidad de los viajes
        for col in cols_identificadores:
            if col in df.columns:
                df_final[col] = df[col].values

        print("Guardando dataset preprocesado...")
        ruta_salida = "data/processed/mibici_pca_v1.csv"
        df_final.to_csv(ruta_salida, index=False)

        # 5. Versionamiento en MLflow
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_param("pca_varianza_retenida", 0.95)
        mlflow.log_metric("features_originales", columnas_originales)
        mlflow.log_metric("features_pca", columnas_nuevas)
        mlflow.log_artifact(ruta_salida, artifact_path="dataset_preprocesado")

        print("¡Preprocesamiento y versionamiento en MLflow completados!")

if __name__ == "__main__":
    preprocesar_datos()