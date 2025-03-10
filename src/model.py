#############################################
# # # 02. MODEL ARQUITECTURE DEFINITION # # #
#############################################
#------------#
# # INDICE # #
#------------#
# 1. Cargue de librerias
# 2. Definicion de la arquitectura

#==================================================================
#--------------------------------#
# # # 1. Cargue de librerias # # #
#--------------------------------#
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

#------------------------------------------#
# # # 2. Definicion de la arquitectura # # #
#------------------------------------------#
def constr_modelo(x_train):
    '''
    Se crea la arquitectura del modelo para su estimacion.
    Argumentos:
        * x_train: Base de entrenamieto; para conocer las dimensiones de la base en la capa de entrada
    Retorno:
        * model: Modelo propuesto con la arquitectura definida
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
                metrics=['mape']
                )
    
    return model
