# Modelo_inmoviliario
Taller DL de predicción del precio inmoviliario usando Backpropagation

Este proyecto desarrolla un modelo de red neuronal backpropagation para predecir el precio de alquiler de viviendas en una ciudad. Se estructura un repositorio incluyendo codigo modular, documentacion clara y ejemplos reproducibles.

Voy a dividir el análisis en los siguientes apartados:

1. Objetivo y Datos Utilizados
2. Exploracion del Dataset
3. Modelo de red neuronal
4. Requisitos previos y dependencias
5. Descripcion de archivos
6. Interactuando con el proyecto
7. Prediciendo desde predict.py
8. Conclusion


## 1. Objetivo y Datos Utilizados

Dado un conjunto de datos con informacion sobre alquileres (metros cuadrados, numero de habitaciones, ubicacion, antiguedad del inmueble, etc.), se requiere predecir el precio mensual. Esta es una problematica real en aplicaciones inmobiliarias y de analisis de mercado.

Este taller utilizara un conjunto de datos que contiene informacion detallada sobre mas de 4700 propiedades residenciales disponibles para alquiler, abarcando casas, apartamentos y pisos. El conjunto de datos incluye las siguientes variables. 

- BHK: Numero de habitaciones, sala y cocina.
- Rent: Precio de alquiler de las casas/apartamentos/pisos.
- Size: Tamano de las casas/apartamentos/pisos en pies cuadrados.
- Floor: Piso en el que se encuentra la propiedad y el total de pisos en el edificio (Ejemplo: Planta baja de 2, 3 de 5, etc.).
- Area Type: Calculo del tamano en Superficie Total, Superficie util o Area Construida.  
- Area Locality: Localidad de la propiedad.
- City: Ciudad donde esta ubicada la propiedad.
- Furnishing Status: Estado de amueblado de la propiedad (Amueblado, Semi-Amueblado o No Amueblado).
- Tenant Preferred: Tipo de inquilino preferido por el dueno o agente.
- Bathroom: Numero de banos.
- Point of Contact: Persona de contacto para obtener mas informacion sobre la propiedad.

## 2. Exploración del DataSet

La variable 'SalePrice' es la variable objetivo de este conjunto de datos. En pasos posteriores a este análisis exploratorio de datos se realizaría una predicción del valor de esta variable, por lo que voy a estudiarla con mayor detenimiento. A simple vista se pueden apreciar:

* Una desviación con respecto a la distribución normal.
* Una asimetría positiva.
* Algunos picos.

## 3. Modelo de Red Neuronal


## 4. Requisitos previos y dependencias

## 5. Descripcion de archivos
(Texto descriptivo de lo que contiene cada archivo o que se realiza en cada uno)

* Notebooks
1) [EDA.ipynb](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/016f7e17d6a2cf41fbee5fc9e3df1485255aad8f/notebooks/00%20EDA.ipynb)
2) [Depuracion.ipynb](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/016f7e17d6a2cf41fbee5fc9e3df1485255aad8f/notebooks/01%20Depuracion.ipynb)
3) [Entrenamiento.ipynb](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/016f7e17d6a2cf41fbee5fc9e3df1485255aad8f/notebooks/02%20Entrenamiento.ipynb)

* src
1) [data_loader.py](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/6cd9ff0a413aca8cf5e9d284076aa98e2ca083c9/src/data_loader.py)
2) [evaluate.py](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/6cd9ff0a413aca8cf5e9d284076aa98e2ca083c9/src/evaluate.py)
3) [model.py](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/6cd9ff0a413aca8cf5e9d284076aa98e2ca083c9/src/model.py)
4) [train.py](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/6cd9ff0a413aca8cf5e9d284076aa98e2ca083c9/src/train.py)
5) [predict.py](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/6cd9ff0a413aca8cf5e9d284076aa98e2ca083c9/src/predict.py)
6) [utils.py](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/6cd9ff0a413aca8cf5e9d284076aa98e2ca083c9/src/utils.py)

* models
1) [model_v1.h5](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/016f7e17d6a2cf41fbee5fc9e3df1485255aad8f/models/model_v1.h5)
2) [predictions_with_data.csv](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/016f7e17d6a2cf41fbee5fc9e3df1485255aad8f/models/predictions_with_data.csv)
3) [scaler.pkl](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/016f7e17d6a2cf41fbee5fc9e3df1485255aad8f/models/scaler.pkl)
4) [test_data.pkl](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/016f7e17d6a2cf41fbee5fc9e3df1485255aad8f/models/test_data.pkl)


## 6. Interactuando con el proyecto
Para la correcta ejecusión de los notebooks se debe realizar lo siguiente:

* Notebooks
1) [EDA.ipynb](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/016f7e17d6a2cf41fbee5fc9e3df1485255aad8f/notebooks/00%20EDA.ipynb)
2) [Depuracion.ipynb](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/016f7e17d6a2cf41fbee5fc9e3df1485255aad8f/notebooks/01%20Depuracion.ipynb)
3) [Entrenamiento.ipynb](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/016f7e17d6a2cf41fbee5fc9e3df1485255aad8f/notebooks/02%20Entrenamiento.ipynb)

* Archivos .py
1) [data_loader.py](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/6cd9ff0a413aca8cf5e9d284076aa98e2ca083c9/src/data_loader.py)
2) [model.py](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/6cd9ff0a413aca8cf5e9d284076aa98e2ca083c9/src/model.py)
3) [train.py](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/6cd9ff0a413aca8cf5e9d284076aa98e2ca083c9/src/train.py)
4) [evaluate.py](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/6cd9ff0a413aca8cf5e9d284076aa98e2ca083c9/src/evaluate.py)
5) [predict.py](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/6cd9ff0a413aca8cf5e9d284076aa98e2ca083c9/src/predict.py)
6) [utils.py](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/6cd9ff0a413aca8cf5e9d284076aa98e2ca083c9/src/utils.py) 


## 7. Rendimiento del modelo



## Conclusión

A lo largo de este proyecto


























