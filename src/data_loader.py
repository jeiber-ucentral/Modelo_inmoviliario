############################################
# # # 01. DATA LOADER AND DATA CLEANER # # # 
############################################
#------------#
# # INDICE # #
#------------#
# 1. Cargue de librerias
# 2. Funcion de cargue de la base de datos
# 3. Funcion de depuracion de los datos
# 4. Funcion consolidada

#==================================================================

#--------------------------------#
# # # 1. Cargue de librerias # # #
#--------------------------------#
import pandas as pd
import numpy as np
import re
import os

#--------------------------------------------------#
# # # 2. Funcion de cargue de la base de datos # # #
#--------------------------------------------------#
def cargue_datos(ruta, msj = True):
  '''
  Funcion que realiza el cargue de la base de datos de acuerdo a la ruta especificada
  Argumentos:
    * Ruta: ruta del archivo a cargar
    * * msj: Por default es True. Indica si se desea impriir un diagnostico de cantidad de filas y primeros registros de la base final

  Retorno:
    * Base de datos cargada en un dataframe
    * Especificaciones de la dimension de registros cargados
    * Impresion de un primer bosquejo de los datos
  '''
  # Cargue de base de datos de acuerdo a ruta dada
  df = pd.read_csv(ruta, sep = ",")

  # Mensajes confirmatorios
  print("\n")
  print("CARGUE EXITOSO DE LA BASE DE DATOS!!" + "\n")
  if msj:
    print(f'***Se han cargado {df.shape[0]} filas y {df.shape[1]} columnas de forma satisfactoria *** \n')
    print(df.head(3))

  return df

#-----------------------------------------------#
# # # 3. Funcion de depuracion de los datos # # #
#-----------------------------------------------#
def depurador_df(df, msj = True):
  '''
  Funcion que realiza la transformacion de la base de datos al formato adecuado para entrenamiento y consumo de los modelos
  Argumentos:
    * df: dataframe con la base de datos a transformar
    * msj: Por default es True. Indica si se desea impriir un diagnostico de cantidad de filas y primeros registros de la base final

  Retorno:
    * Base de datos transformada en un dataframe
    * Especificaciones de la dimension de registros depurados y primeros registros si se desea
  '''

  # # # Ajuste variable floor
  df['residence_floor'] = df['Floor'].str.split(' out of ', expand=True)[0]
  df['residence_max_floor'] = df['Floor'].str.split(' out of ', expand=True)[1]

  # Con nombre de piso
  df['residence_floor'] = df['residence_floor'].apply(lambda x: re.sub(r'Ground', '1', x))         # Ground se vuelve el piso 1
  df['residence_floor'] = df['residence_floor'].apply(lambda x: re.sub(r'Lower Basement', '1', x)) # Lower Basement se vuelve el piso 1
  df['residence_floor'] = df.apply(lambda row: re.sub(r'Upper Basement', row['residence_max_floor'], row['residence_floor']) if row['residence_floor'] == 'Upper Basement' else row['residence_floor'], axis=1)
  df['residence_max_floor'] = df.apply(lambda row: row['residence_floor'] if pd.isnull(row['residence_max_floor']) else row['residence_max_floor'], axis=1)

  # Convertir a numericas
  df['residence_floor'] = df['residence_floor'].astype(int)
  df['residence_max_floor'] = df['residence_max_floor'].astype(int)

  # # Revision variable Tenant Preferred
  df["Tenant_bachelors"] = df["Tenant Preferred"].apply(lambda x: 1 if "Bachelors" in x else 0)
  df["Tenant_family"] = df["Tenant Preferred"].apply(lambda x: 1 if "Family" in x else 0)

  # # Selección de variables
  df = df[['Rent',	'BHK',	'Size',	'Area Type',	'City',	'Furnishing Status',	'Bathroom',	'Point of Contact',	'residence_floor',	'residence_max_floor',	'Tenant_bachelors',	'Tenant_family']]

  # Dicotomizando caracteristicas tipo object
  df = pd.get_dummies(df, columns=['Area Type', 'City', 'Furnishing Status', 'Point of Contact'], drop_first=True).astype(int)
  
  # Mensaje de transformacion exitosa
  print("SE HA DEPURADO LA BASE DE DATOS!!" + "\n")
  if msj:
    print(f'***Dimensiones finales {df.shape[0]} filas y {df.shape[1]} columnas *** \n')
    print(df.head(3))

  return df


#--------------------------------#
# # # 4. Funcion consolidada # # #
#--------------------------------#
def main(ruta, msj = False):
  '''
  Funcion que realiza todo el proceso de cargue y depuracion de la base de datos
  # Argumentos:
    * ruta: ruta del archivo a cargar
    * mensajes: Por default es True. Indica si se desea impriir un diagnostico de cantidad de filas y primeros registros de la base final
  
  # Retorno:
    * base de datos en formato para entrenamiento y consumo del modelo
    * Mensajes con cantidad de registros cargados, dimensiones y primeros registros de ejemplo (si se desea)
  
  '''

  # Cargue de base de datos
  df = cargue_datos(ruta, msj = msj)

  print("\n")

  # Depuracion de la base
  df_adj = depurador_df(df, msj = msj)

  return df_adj


if __name__ == '__main__':
    ruta = input("Ingresar la ruta del archivo House_Rent_Dataset.csv (sin comillas): ").strip()
    msj = input("Desea ver detalles de las bases resultantes? (True / False): ").strip().lower() in ["true", "1", "yes"]

    if not os.path.exists(ruta):
        print("⚠️ La ruta ingresada no es válida. Verifica e intenta nuevamente.")
    else:
        main(ruta=ruta, msj=msj)

