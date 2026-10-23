import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import mlflow

class MiBiciPreprocessor:
    """
    Clase para manejar el preprocesamiento de datos de MiBici 
    aplicando Programación Orientada a Objetos (POO).
    """
    def __init__(self, varianza_pca=0.95, random_state=42):
        # El constructor inicializa las herramientas que el objeto usará
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=varianza_pca, random_state=random_state)
        self.columnas_id = ['Viaje_Id', 'Usuario_Id', 'Origen_Id', 'Destino_Id']
        
    def cargar_datos(self, filepath):
        print(f"Cargando datos desde {filepath}...")
        return pd.read_csv(filepath, low_memory=False)
        
    def limpiar_columnas(self, df, columnas_a_eliminar):
        print("Limpiando columnas vacías o innecesarias...")
        return df.drop(columns=[col for col in columnas_a_eliminar if col in df.columns])
        
    def aplicar_transformaciones(self, df):
        print("Aplicando Normalización y Reducción de Dimensionalidad (PCA)...")
        # Aislar variables matemáticas de los IDs
        df_features = df.drop(columns=[col for col in self.columnas_id if col in df.columns])
        df_features = df_features.fillna(df_features.median())
        
        # Ejecutar transformaciones (usando los atributos inicializados en el constructor)
        datos_escalados = self.scaler.fit_transform(df_features)
        datos_pca = self.pca.fit_transform(datos_escalados)
        
        # Reconstruir el DataFrame
        columnas_pca = [f"PC{i+1}" for i in range(datos_pca.shape[1])]
        df_final = pd.DataFrame(datos_pca, columns=columnas_pca)
        
        # Reintegrar los identificadores
        for col in self.columnas_id:
            if col in df.columns:
                df_final[col] = df[col].values
                
        return df_final, df_features.shape[1], datos_pca.shape[1]

    def ejecutar_pipeline(self, input_path, output_path):
        """
        Método principal que orquesta el flujo completo y registra en MLflow.
        """
        mlflow.set_experiment("Proyecto_MiBici_POO")
        with mlflow.start_run(run_name="Refactorizacion_Clases"):
            df = self.cargar_datos(input_path)
            df = self.limpiar_columnas(df, ['id', 'latitude', 'longitude', 'dpcapacity'])
            
            df_final, cols_orig, cols_pca = self.aplicar_transformaciones(df)
            
            print(f"Guardando resultados refactorizados en {output_path}...")
            df_final.to_csv(output_path, index=False)
            
            # Registro de versionado de datos en MLflow
            mlflow.log_param("paradigma", "Programación Orientada a Objetos")
            mlflow.log_metric("features_originales", cols_orig)
            mlflow.log_metric("features_pca", cols_pca)
            mlflow.log_artifact(output_path, artifact_path="dataset_poo")
            
            print("¡Pipeline orientado a objetos ejecutado con éxito!")

# Bloque de ejecución principal
if __name__ == "__main__":
    # Instanciar el objeto y ejecutar el proceso
    preprocesador = MiBiciPreprocessor()
    preprocesador.ejecutar_pipeline(
        input_path='data/processed/mibici_modelado_v1.csv',
        output_path='data/processed/mibici_refactorizado.csv'
    )