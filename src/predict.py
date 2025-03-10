####################################
# # # 05. USE MODEL TO PREDICT # # # 
####################################
#------------#
# # INDICE # #
#------------#
# 1. Cargue de librerias
# 2. Cargue del modelo
# 3. Funcion de prediccion
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
import data_loader 

#--------------------------------#
# # # # 2. Cargue del modelo # # #
#--------------------------------#
def cargar_modelo():
    '''
    Carga el modelo guardado desde la carpeta models.
    Argumentos: 

    Retorno:
        - model: Modelo cargado.
    '''
    # Cargue del modelo
    model = tf.keras.models.load_model("models/model_v1.h5")
    escalador = joblib.load("models/scaler.pkl")
    print("Modelo cargado correctamente 👌")

    return model, escalador

#----------------------------------#
# # # 3. Funcion de prediccion # # #
#----------------------------------#
def prediccion_nueva_observacion(modelo, f_escalado, registros):
    # columnas faltantes dada la actividad de datos dummies
    columnas = ['BHK', 'Size', 'Bathroom', 'residence_floor',
                'residence_max_floor', 'Tenant_bachelors', 'Tenant_family',
                'Area Type_Carpet Area', 'Area Type_Super Area', 'City_Chennai',
                'City_Delhi', 'City_Hyderabad', 'City_Kolkata', 'City_Mumbai',
                'Furnishing Status_Semi-Furnished', 'Furnishing Status_Unfurnished',
                'Point of Contact_Contact Builder', 'Point of Contact_Contact Owner']
    col_faltantes = set(columnas) - set(registros.columns)  
    for col in col_faltantes:
        registros[col] = 0  # or any suitable default value

    # estructura de la bd para modelo
    if 'Rent' in registros.columns:
        registros = registros.drop(columns=['Rent'])

    # Estandarizar la nueva observación
    registros_escalados = f_escalado.transform(registros)

    # Hacer la prediccion
    prediction = modelo.predict(registros_escalados)

    # Agregar predicciones a los datos originales
    registros_or = registros.copy()
    registros_or["Predicted_Rent"] = prediction.flatten()

    return registros_or

#--------------------------------#
# # # 4. Funcion consolidada # # #
#--------------------------------#
def main(ruta, msj = False):
    '''
    Ejecuta el proceso de estandarizacion y estimacion de los datos en formato original para su prediccion de precio de renta
    Argumentos: 
        * ruta: ruta del archivo a cargar para estimar precio de renta
        * mensajes: Por default es True. Indica si se desea imprimir un diagnostico de cantidad de filas y primeros registros de la base final
    Retorno:
        * Prediccion: retorna la prediccion del precio de renta dados los nuevos datos
        * archivo: Archivo csv con la prediccion de ser requerido
    '''
    # Cargue del modelo
    model, escalador = cargar_modelo()
    
    # Cargue de los datos nuevos
    datos = data_loader.cargue_datos(ruta, msj = msj)

    # Depuracion de la base
    df_adj = data_loader.depurador_df(datos)

    # prediccion de la(s) nueva(s) observacion(es)
    df_prediccion = prediccion_nueva_observacion(model, escalador, df_adj)

    df_prediccion.to_csv("models/predictions_with_data.csv", index=False)
    print("Predicciones guardadas en 'models/predicciones.csv' 👌")

    return df_prediccion


if __name__ == '__main__':
    ruta = input("Ingrese la ruta del archivo CSV con nuevos datos: ").strip()

    main(ruta)




