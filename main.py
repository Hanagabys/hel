# ====================================
# 📊 11. Análisis de Importancia de Variables (Feature Importance)
# ====================================
import matplotlib.pyplot as plt
import pickle
import os
# Intentar cargar el modelo optimizado
if os.path.exists('best_rf_model.pkl'):
    with open('best_rf_model.pkl', 'rb') as file:
        best_rf = pickle.load(file)
    print('Modelo cargado exitosamente.')
else:
    print('Error: El archivo best_rf_model.pkl no se encuentra.')
    best_rf = None
# Verificar si el modelo fue cargado correctamente
if best_rf:
    # Extraer la importancia de las variables del modelo optimizado
    feature_importances = best_rf.feature_importances_
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    # Crear un DataFrame con los resultados
    import pandas as pd
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    # Mostrar las 10 variables más importantes
    print("\nImportancia de Variables (Top 10):")
    print(importance_df.head(10))
    # ====================================
    # 📈 12. Gráfico de Correlación
    # ====================================
    import seaborn as sns
    plt.figure(figsize=(12, 10))
    correlation_matrix = importance_df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Mapa de Correlación de Variables')
    plt.show()
else:
    print('No se pudo realizar el análisis de importancia de variables debido a la falta del modelo.')
