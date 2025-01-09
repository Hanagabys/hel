# ====================================
# 📊 11. Análisis de Importancia de Variables (Feature Importance)
# ====================================
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
# Cargar el modelo optimizado desde el archivo
if os.path.exists('best_rf_model.pkl'):
    with open('best_rf_model.pkl', 'rb') as file:
        best_rf = pickle.load(file)
    print('Modelo cargado exitosamente.')
else:
    print('Error: El archivo best_rf_model.pkl no se encuentra.')
    best_rf = None
# Guardar nuevamente el modelo con la versión correcta de scikit-learn
if best_rf:
    with open('best_rf_model.pkl', 'wb') as file:
        pickle.dump(best_rf, file)
    print('Modelo guardado con la versión compatible de scikit-learn.')
