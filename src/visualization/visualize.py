import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def realizar_eda():
    print("Cargando dataset para el Análisis Exploratorio...")
    # Usamos el dataset limpio (antes del One-Hot Encoding) para que las gráficas 
    # categóricas como el género sean legibles
    df_limpio = pd.read_csv('data/processed/mibici_limpio_v1.csv', low_memory=False)
    
    # Crear la carpeta de reportes si no existe
    os.makedirs('reports/figures', exist_ok=True)

    print("\n--- Estadísticas Descriptivas ---")
    # Genera conteos, medias, desviaciones estándar, mínimos y máximos
    estadisticas = df_limpio.describe()
    print(estadisticas)
    
    # Guardar las estadísticas en un archivo de texto para documentar la tarea
    with open('reports/estadisticas_descriptivas.txt', 'w') as f:
        f.write("Estadísticas Descriptivas de MiBici\n")
        f.write("=====================================\n")
        f.write(estadisticas.to_string())

    print("\nGenerando visualizaciones...")
    sns.set_theme(style="whitegrid")

    # 1. Histograma: Distribución de la duración de los viajes
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_limpio, x='duracion_min', bins=50, kde=True, color='skyblue')
    plt.title('Distribución de la Duración de Viajes (minutos)')
    plt.xlabel('Duración (min)')
    plt.ylabel('Frecuencia')
    plt.savefig('reports/figures/1_distribucion_duracion.png')
    plt.close()

    # 2. Gráfico de barras: Cantidad de viajes por Género
    if 'Genero' in df_limpio.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df_limpio, x='Genero', palette='Set2')
        plt.title('Cantidad de Viajes por Género')
        plt.savefig('reports/figures/2_viajes_por_genero.png')
        plt.close()

    # 3. Mapa de calor (Heatmap): Matriz de correlación
    # Para la correlación, necesitamos el dataset donde ya convertiste todo a números
    print("Generando matriz de correlación con el dataset modelado...")
    df_num = pd.read_csv('data/processed/mibici_modelado_v1.csv', low_memory=False)
    
    plt.figure(figsize=(12, 8))
    matriz_corr = df_num.corr()
    # Usamos cmap coolwarm para ver correlaciones positivas en rojo y negativas en azul
    sns.heatmap(matriz_corr, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Matriz de Correlación de Variables Numéricas')
    plt.savefig('reports/figures/3_matriz_correlacion.png')
    plt.close()

    print("\n¡EDA completado! Revisa la carpeta 'reports/figures/' para ver tus gráficas.")

if __name__ == "__main__":
    realizar_eda()