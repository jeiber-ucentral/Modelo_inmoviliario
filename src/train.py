###########################
# # # 03. TRAIN MODEL # # # 
###########################

#------------#
# # INDICE # #
#------------#
# 1. Cargue de librerias
# 2. Cargue y Segmentacion de datos
# 3. Entrenamiento del modelo
# 4. Exportar el modelo, scaler y datos de prueba
# 5. Funcion consolidada

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

from model import constr_modelo  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

import tensorflow as tf

#-------------------------------------------#
# # # 2. Cargue y Segmentacion de datos # # #
#-------------------------------------------#
def division_datos(ruta, mensajes=True):
    '''
    Carga y segmenta los datos en entrenamiento, validacion y prueba.
    Guarda el scaler para normalizar datos en la prediccion.
    Argumentos:
        * ruta: ruta del archivo a cargar
        * mensajes: Por default es True. Indica si se desea imprimir un diagnostico de cantidad de filas y primeros registros de las bases finales

    Retorno:
        * x_train, x_test, x_val
        * y_train, y_test, y_val
        * scaler utilizado para normalizar los datos
    '''
    df = data_loader.main(ruta=ruta, msj=mensajes)

    # Division en variables predictoras (X) y objetivo (Y)
    x = df.drop('Rent', axis=1)
    y = df['Rent']

    # Escalado de datos
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # Division en train, test y validacion
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=42)

    if mensajes:
        print("x_train: ", x_train.shape)
        print("x_test: ", x_test.shape)
        print("x_val: ", x_val.shape)
    
    return x_train, x_test, x_val, y_train, y_test, y_val, scaler

#-------------------------------------#    
# # # 3. Entrenamiento del modelo # # #
#-------------------------------------#
def entrenamiento(x_train, x_val, y_train, y_val, grafico=True):
    '''
    Entrena el modelo y muestra el grafico de desempeno (si se desea).
    Argumentos:
        * x_train : Datos de entrenamiento ; entradas del modelo para el entrenamiento
        * x_val : Datos de validacion del modelo ; entradas del modelo para la validacion
        * y_train  : Datos de entrenamiento ; salida del modelo para el entrenamiento
        * y_val  : Datos de validacion ; salida del modelo para la validacion 
        * grafico : (True or False) si se desea (True) mostrar el grafico de desempeno del modelo en cuanto a funcion de perdida y metrica por epoca
    Retorno:
        * model: Retorna el modelo estimado de acuerdo a la arquitectura dada y los datos de entrenamiento y validacion
        * History: Historia del proceso de estimacion del modelo por epoca (funcion de perdida y metricas)
        * Grafico de desempeno (si se desea)
    '''
    # Cargar la arquitectura del modelo
    model = constr_modelo(x_train = x_train)

    # Entrenando el modelo
    history = model.fit(x_train, y_train,
                        epochs=50,
                        batch_size=16,
                        verbose=0,
                        validation_data=(x_val, y_val)
                        )

    if grafico:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Grafico de MAPE
        axes[0].plot(history.history['mape'])
        axes[0].plot(history.history['val_mape'])
        axes[0].set_title('Model MAPE')
        axes[0].set_ylabel('MAPE')
        axes[0].set_xlabel('Epoch')
        axes[0].legend(['Train', 'Validation'], loc='upper left')

        # Grafico de perdida (Loss)
        axes[1].plot(history.history['loss'])
        axes[1].plot(history.history['val_loss'])
        axes[1].set_title('Model Loss')
        axes[1].set_ylabel('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].legend(['Train', 'Validation'], loc='upper left')

        plt.tight_layout()
        plt.show()
    
    return model, history

#----------------------------------------#
# # # 4. Exportar el modelo y scaler # # #
#----------------------------------------#
def exportar_modelo(model, scaler, x_test, y_test):
    '''
    Guarda el modelo entrenado, el scaler y los datos de prueba.
    
    Argumentos:
        * model: Modelo estimado dada la arquitectura
        * scaler: Funcion de escalado de datos empleada en la base train y validation (para funcion predict)
        * x_test: base de test ; entradas del modelo para el testeo del modelo
        * y_test base de test ; salidas del modelo para el testeo del modelo

    Retorno:
        * Guardado del modelo, funcion scaler y datos de prueba en la carpeta models
    '''
    # Guardar el modelo estimado
    modelo_path = "models/model_v1.h5"
    model.save(modelo_path)
    print(f"Modelo guardado en {modelo_path} 👌")

    # Guardar la funcion de escalado empleada
    scaler_path = "models/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler guardado en {scaler_path} 👌")

    # Guardar datos de prueba para validaciones
    test_data_path = "models/test_data.pkl"
    joblib.dump((x_test, y_test), test_data_path)
    print(f"Datos de prueba guardados en {test_data_path} 👌")

#--------------------------------#
# # # 5. Funcion consolidada # # #
#--------------------------------#
def main(ruta, msj=True, grafico=True):
    '''
    Ejecuta todo el proceso de carga, entrenamiento y exportacion de archivos.
    Argumentos:
        * ruta: ruta del archivo a cargar
        * msj: Por default es True. Indica si se desea imprimir un diagnostico de cantidad de filas y primeros registros de las bases finales
        * grafico : (True or False) si se desea (True) mostrar el grafico de desempeno del modelo en cuanto a funcion de perdida y metrica por epoca
    Retorno:
        * Modelo, funcion de escalado de datos y datos de testeo exportados
    '''
    # Cargar y segmentar datos
    x_train, x_test, x_val, y_train, y_test, y_val, scaler = division_datos(ruta, mensajes=msj)

    # Entrenar el modelo
    model, history = entrenamiento(x_train, x_val, y_train, y_val, grafico=grafico)
    print("MODELO ENTRENADO SATISFACTORIAMENTE!! 👌")

    # Guardar el modelo, scaler y datos de prueba
    exportar_modelo(model, scaler, x_test, y_test)

    return model, scaler


if __name__ == '__main__':
    ruta = input("Ingresar la ruta del archivo House_Rent_Dataset.csv (sin comillas): ").strip()
    msj = input("Desea ver detalles de las bases resultantes? (True / False): ").strip().lower() in ["true", "1", "yes"]
    grafico = input("Desea ver el grafico de entrenamiento del modelo? (True / False): ").strip().lower() in ["true", "1", "yes"]

    if not os.path.exists(ruta):
        print("⚠️ La ruta ingresada no es valida. Verifica e intenta nuevamente.")
    else:
        main(ruta, msj, grafico)
