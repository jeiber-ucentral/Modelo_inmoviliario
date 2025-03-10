##############################
# # # 04. EVALUATE MODEL # # # 
##############################
#------------#
# # INDICE # #
#------------#
# 1. Cargue de librerias
# 2. Cargue del modelo y los datos de testeo
# 3. Funcion de evaluacion del modelo
# 4. Funcion consolidada

#==================================================================

#--------------------------------#
# # # 1. Cargue de librerias # # #
#--------------------------------#
import warnings
warnings.filterwarnings("ignore")

import os
import joblib 
import numpy as np
import pandas as pd
import data_loader
import matplotlib.pyplot as plt
  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

import tensorflow as tf

#----------------------------------------------------#
# # # 2. Cargue del modelo y los datos de testeo # # #
#----------------------------------------------------#
def cargar_modelo_y_datos():
    '''
    Carga el modelo guardado y los datos de test desde la carpeta models.
    Argumentos: 

    Retorno:
        - model: Modelo cargado.
        - x_test, y_test: Datos de prueba.
    '''
    # Cargue del modelo
    model = tf.keras.models.load_model("models/model_v1.h5")
    print("Modelo cargado correctamente 👌")

    # Cargue de datos test
    x_test, y_test = joblib.load("models/test_data.pkl")
    print(f"Datos de test cargados: {x_test.shape} registros 👌.")

    return model, x_test, y_test


#---------------------------------------------#
# # # 3. Funcion de evaluacion del modelo # # #
#---------------------------------------------#
def evaluar_modelo(model, x_test, y_test):
    '''
    Realiza la evaluacion del modelo en cuanto a metricas de desempeno y error
    Argumentos:

    Retorno: 
        * resultados: metricas de desempeno del modelo en formato df

    '''

    # Uso del modelo para predecir sobre base test
    y_pred = model.predict(x_test)
    # Calcular métricas de evaluación
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Guardar resultados en df
    results = pd.DataFrame({
        'MSE': [mse],
        'RMSE': [rmse],
        'MAE': [mae],
        'MAPE': [mape],
        'R2': [r2]
    })

    return results


#--------------------------------#
# # # 4. Funcion consolidada # # #
#--------------------------------#
def main():
    '''
    Ejecuta el proceso de validacion del modelo sobre la base de testeo
    Argumentos: 

    Retorno:
        * evaluacion: retorna la evaluacion del modelo en diferentes metricas en formato df
    '''
    model, x_test, y_test = cargar_modelo_y_datos()
    
    evaluacion = evaluar_modelo(model, x_test, y_test)
    print(evaluacion)

    return evaluacion

if __name__ == '__main__':
    main()

