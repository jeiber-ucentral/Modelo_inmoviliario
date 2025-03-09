###########################
# # # 02. TRAIN MODEL # # # 
###########################
#------------#
# # INDICE # #
#------------#
# 1. Cargue de librerias
# 2. Cargue y Segmentacion de datos
# 3. Entrenamiento del modelo
# 4. Exportar el modelo y scaler
# 5. Funcion consolidada

#==================================================================

#--------------------------------#
# # # 1. Cargue de librerias # # #
#--------------------------------#
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import re
import os
import data_loader 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

#-------------------------------------------#
# # # 2. Cargue y Segmentacion de datos # # #
#-------------------------------------------#
def division_datos(ruta, mensajes = True):
    '''
    Realiza el cargue y segmentacion de los datos depurados para entrenamiento, validacion y prueba del modelo,
    asi mismo retorna la funcion de escalado empleada en los datos

    Argumentos:
        *mensajes: Si es verdadero retorna la cantidad de registros que estarán en los datos de entrenamiento, prueba y validacion
    Retorno:
        * Base segmentada en entrenamiento, prueba y validacion tanto de X como de Y (6 bases)
        * Funcion de escalado encontrada
    '''

    # Cargue de los datos limpios
    df = data_loader.main(ruta=ruta, msj=mensajes)

    # Division de los datos en objetivo y regresoras
    x = df.drop('Rent', axis=1)
    y = df['Rent']

    # Estandarizacion de datos
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # # Division de los datos en entrenamiento, test y validacion
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=42)

    if mensajes:
        print("x_train: ",  x_train.shape)
        print("x_test: ", x_test.shape)
        print("x_val: ", x_val.shape)
    
    return x_train, x_test, x_val, y_train, y_test, y_val, scaler

#-------------------------------------#    
# # # 3. Entrenamiento del modelo # # #
#-------------------------------------#
def entrenamiento(x_train, x_test, x_val, y_train, y_test, y_val, grafico = True):
    '''
    '''
    # # Definicion de la Arquitectura
    model = keras.Sequential([
        layers.Dense(128, activation='relu', kernel_regularizer = regularizers.l1_l2(l1 = 0.001, l2 = 0.001), input_shape=(x_train.shape[1],)), 
        layers.Dense(64, activation='relu', kernel_regularizer = regularizers.l1_l2(l1 = 0.001, l2 = 0.001)),  
        layers.Dense(32, activation='relu', kernel_regularizer = regularizers.l1_l2(l1 = 0.001, l2 = 0.001)),    
        layers.Dense(16, activation='relu'),  
        layers.Dense(8, activation='relu'),  
        layers.Dense(1, activation='linear')  
    ])

    # Compilar el modelo
    model.compile(optimizer='adam',  # RMSprop, adam
                loss='mean_squared_error',  
                metrics=['mape'])  

    # Entrenando el modelo
    history = model.fit(x_train, y_train,
                        epochs = 50,
                        batch_size = 16,
                        verbose = 0,
                        validation_data = (x_val, y_val)
                        )

    if grafico:
        # Plot MAPE on the first subplot
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(history.history['mape'])
        axes[0].plot(history.history['val_mape'])
        axes[0].set_title('Model MAPE')
        axes[0].set_ylabel('MAPE')
        axes[0].set_xlabel('Epoch')
        axes[0].legend(['Train', 'Validation'], loc='upper left')

        #Plot Loss on the second subplot
        axes[1].plot(history.history['loss'])
        axes[1].plot(history.history['val_loss'])
        axes[1].set_title('Model Loss')
        axes[1].set_ylabel('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].legend(['Train', 'Validation'], loc='upper left')

        plt.tight_layout()  # Adjusts subplot parameters for a tight layout
        plt.show()
    
    return model

#----------------------------------------#
# # # 4. Exportar el modelo y scaler # # #
#----------------------------------------#
'''
def exportador_procesos(model, scaler):


'''


#--------------------------------#
# # # 5. Funcion consolidada # # #
#--------------------------------#
def main(ruta, msj, grafico):
    # Segmentando la base de datos
    x_train, x_test, x_val, y_train, y_test, y_val, scaler = division_datos(ruta, mensajes = msj)

    # Entrenando el modelo
    model = entrenamiento(x_train, x_test, x_val, y_train, y_test, y_val, grafico = grafico)
    print("MODELO ENTRENADO SATISFACTORIAMENTE!!👌")

    return model, scaler


if __name__ == '__main__':
    ruta = input("Ingresar la ruta del archivo House_Rent_Dataset.csv (sin comillas): ").strip()
    msj = input("Desea ver detalles de las bases resultantes? (True / False): ").strip().lower() in ["true", "1", "yes"]
    grafico = input("Desea ver el grafico de entrenamiento del modelo? (True / False): ").strip().lower() in ["true", "1", "yes"]

    if not os.path.exists(ruta):
        print("⚠️ La ruta ingresada no es válida. Verifica e intenta nuevamente.")
    else:
        main(ruta, msj, grafico)

