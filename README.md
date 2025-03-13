# Modelo_inmoviliario
Taller DL de predicción del precio inmoviliario usando Backpropagation

Este proyecto desarrolla un modelo de red neuronal backpropagation para predecir el precio de alquiler de viviendas en una ciudad. Se estructura un repositorio incluyendo codigo modular, documentacion clara y ejemplos reproducibles.

Se dividirá el análisis en los siguientes apartados:

1. Objetivo y Datos Utilizados
2. Exploracion del Dataset
3. Modelo de red neuronal
4. Requisitos previos y dependencias
5. Descripcion de archivos
6. Interactuando con el proyecto
7. Rendimiento del proyecto
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

En esta sección se describe el proceso de exploración realizado sobre el conjunto de datos utilizado en este proyecto. La exploración de datos es un paso fundamental para comprender su estructura, identificar posibles problemas o patrones, y guiar las decisiones de análisis y modelado. A continuación se detalla cómo se llevó a cabo la exploración y los resultados más relevantes.

### Cargar el Dataset
Se comienza cargando el conjunto de datos en el entorno de trabajo. Se utilizaron las siguientes bibliotecas y métodos para importar los datos:

Librerías usadas: pandas, numpy, matplotlib, seaborn

Método de carga: pd.read_csv() para cargar el archivo CSV. 

### Inspección Inicial
Se realizaron las siguientes acciones para obtener una visión preliminar del conjunto de datos:

Visualización de las primeras filas con df.head() y df.columns().

Análisis de las estadísticas descriptivas de la variable objetivo "Rent" con df.describe(), para obtener un resumen numérico de la variable.


### Análisis Exploratorio de Datos (EDA)
Se realizaron análisis más profundos para entender mejor las relaciones entre las variables:

Distribución de variables: Se analizaron las distribuciones de las variables numéricas y categóricas a través de gráficos como histogramas y diagramas de cajas (boxplots) para ver outliers.

Se ejecutaron las fuciones de asimetria y curtosis para determinar el sesgo en las variables y sus posibles valores extremos.

Valores atípicos: Se identificaron posibles outliers que pudieran afectar los modelos y se tratarlos por medio de transformación logaritmica. 

Se estandarizan los datos para conviertir los valores de a una escala común, con media = 0 y desviación estándar = 1.

Correlaciones: Se calculó la matriz de correlación para identificar posibles relaciones entre variables numéricas.

Relaciones entre variables: Se utilizaron gráficos como diagramas de dispersión (scatter plots) para observar las relaciones entre las variables.


### Limpieza de Datos
Se identificaron y gestionaron posibles problemas con los datos:

Valores nulos: Se identificaron columnas con valores faltantes y se encuentra que no hay nulos

Se verificó la presencia de normalidad en las distibuciones


### Visualización de Datos
Se utilizaron diversas visualizaciones para presentar los patrones encontrados:

Histogramas y diagramas de dispersión.

Gráficos de calor (heatmaps) para visualizar las correlaciones entre las variables.

Diagramas de caja para mostrar la variabilidad y la presencia de valores atípicos.


### Conclusiones Iniciales
A partir de la exploración de los datos, se extrajeron las siguientes conclusiones clave:

*Se evidencia una distribucion sesgada a la derecha (sesgo positivo), con mayoría de valoes concentrados hacia el 0. Cantidad significativa de outliers a la derecha indicando valores de renta muy altos.

*La renta tiene una alta correlacion con variables como tamaño, y cantidad de baños. 

*De los gráficos de histograma se evidencia que la matoría de las variables no sigue una distribución normal. 


## 3. Modelo de Red Neuronal

En este proyecto, se implementó una **red neuronal de retropropagación (backpropagation)** utilizando varias bibliotecas de Python, como **Pandas**, **NumPy**, **Matplotlib** y **TensorFlow/Keras**. El objetivo del modelo es predecir el precio de alquiler de viviendas basado en características como el tamaño de la propiedad, la ubicación, el número de habitaciones, el estado de amueblado, entre otras.
El modelo consta de las siguientes capas:
 **Capa de Entrada**:
   - Número de neuronas igual al número de características del dataset después del preprocesamiento. Estas características incluyen variables numéricas (como el tamaño de la propiedad y el número de habitaciones) y variables categóricas codificadas (como la ubicación y el estado de amueblado).
**Capas Ocultas**:
   - **Primera Capa Oculta**: 64 neuronas con función de activación **ReLU** (Rectified Linear Unit). Esta capa captura relaciones no lineales entre las características.
   - **Segunda Capa Oculta**: 32 neuronas con función de activación **ReLU**. Esta capa ayuda a refinar las características aprendidas por la primera capa.
   - **Dropout**: Se aplicó una tasa de dropout del 50% (0.5) después de cada capa oculta para evitar el sobreajuste (overfitting). El dropout desactiva aleatoriamente neuronas durante el entrenamiento, lo que mejora la generalización del modelo.
**Capa de Salida**:
   - 1 neurona con función de activación **lineal**. Dado que este es un problema de regresión (predicción de un valor continuo, como el precio de alquiler), la capa de salida no utiliza una función de activación no lineal.


## 4. Requisitos previos y dependencias

## 5. Descripcion de archivos
(Texto descriptivo de lo que contiene cada archivo o que se realiza en cada uno)

* Notebooks
1) [EDA.ipynb](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/016f7e17d6a2cf41fbee5fc9e3df1485255aad8f/notebooks/00%20EDA.ipynb): Este archivo de Python está diseñado para llevar a cabo un análisis exploratorio de datos (EDA) del dataset propuesto. El flujo de trabajo sigue una serie de pasos para explorar, visualizar y preparar los datos para un análisis más profundo o modelado predictivo. Dichos pasos principales incluyen:
- Cargar y mostrar los primeros datos.
- Análisis descriptivo.
- Visualización de la distribución de los datos.
- Transformaciones de los datos.
- Análisis bivariado.
- Análisis de datos categóricos.
- Análisis de valores faltantes.
- Gráficos de probabilidad normal.
- Es un archivo completo de análisis exploratorio de datos que realiza tareas de carga, limpieza, visualización, transformación, y análisis estadístico. Su objetivo principal es preparar y comprender mejor los datos para futuros análisis o modelados predictivos. Las visualizaciones y estadísticas proporcionadas permiten entender la distribución de las variables, la presencia de valores atípicos, la relación entre variables y la calidad de los datos.
2) [Depuracion.ipynb](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/016f7e17d6a2cf41fbee5fc9e3df1485255aad8f/notebooks/01%20Depuracion.ipynb): Con este archivo se realiza un proceso integral de limpieza de datos del dataset, entre las principales tareas se incluye la creación de nuevas variables, la transformación de variables categóricas en variables binarias, la corrección de errores en las columnas y la eliminación de datos irrelevantes. Al final del proceso, se guarda el DataFrame limpio y preparado para análisis posteriores.
3) [Entrenamiento.ipynb](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/016f7e17d6a2cf41fbee5fc9e3df1485255aad8f/notebooks/02%20Entrenamiento.ipynb): Es un notebook que contiene el proceso para entrenar la red neuronal de regresión utilizando el dataset de alquileres de casas. Incluye la carga y preprocesamiento de los datos, la definición y entrenamiento del modelo de red neuronal, la visualización de las métricas de entrenamiento y la evaluación final del modelo en el conjunto de prueba.

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
Para empezar, para la correcta ejecución de los procesos se debe clonar el repositorio a su equipo local empleando GIT, siendo el link del repositorio el siguiente: https://github.com/jeiber-ucentral/Modelo_inmoviliario.git

Una vez clonado el repositorio se debe abrir el folder del proyecto mediante Visual Studio Code o su editor de código preferido.

Así mismo, es necesario contar localmente con el archivo "**House_Rent_Dataset.csv"**. Este no se presenta en el repositorio como un insumo, por lo que se asume se encuentra en su maquina local.

Para la correcta ejecución de los notebooks se debe realizar lo siguiente:

* Notebooks
1) [EDA.ipynb](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/016f7e17d6a2cf41fbee5fc9e3df1485255aad8f/notebooks/00%20EDA.ipynb)
2) [Depuracion.ipynb](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/016f7e17d6a2cf41fbee5fc9e3df1485255aad8f/notebooks/01%20Depuracion.ipynb)
3) [Entrenamiento.ipynb](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/016f7e17d6a2cf41fbee5fc9e3df1485255aad8f/notebooks/02%20Entrenamiento.ipynb)

* Archivos .py
1) [data_loader.py](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/6cd9ff0a413aca8cf5e9d284076aa98e2ca083c9/src/data_loader.py)

Al ejecutar en la terminal mediante el comando **python src/data_loader.py** en la terminal saldrá un mensaje que pedirá escribir la ruta del archivo "**House_Rent_Dataset.csv"**. Se debe tener a la mano la ruta donde se aloja este archivo de forma local para posteriormente pegarlo en dicha variable de entrada manual. NOTA:// Usualmente al copiar la ruta esta al pegarla sale entre comillas. De ser el caso estas deben ser omitidas dentro de la ruta eviada. 

Acto seguido el sistema pedira determinar si se desea ver los detalles del cargue de los datos. En caso de ser así y querer ver este status de cargue de datos se debe digitar la palabra **"True"**. Caso contrario **"False"**. Una vez diligenciados estos requerimientos el proceso ejecutara  en pantalla mostrara el resultado obtenido, siendo este el cargue satisfactorio de la base de datos y su depuración para el proceso de entrenamiento del modelo propuesto y su uso.

2) [model.py](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/6cd9ff0a413aca8cf5e9d284076aa98e2ca083c9/src/model.py)

Este proceso define la arquitectura de la red neuronal backpropagation propuesta para la estimación del valor de la renta de un hogar. En este caso el código, si bien es ejecutable no retornará nada. Este modulo es empleado en el script de entrenamiento del modelo, por tanto no es necesaria su ejecución.

3) [train.py](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/6cd9ff0a413aca8cf5e9d284076aa98e2ca083c9/src/train.py)

Al ejecutar en la terminal mediante el comando **python src/train.py** en la terminal saldrá un mensaje que pedirá escribir la ruta del archivo "**House_Rent_Dataset.csv"**. Se debe tener a la mano la ruta donde se aloja este archivo de forma local para posteriormente pegarlo en dicha variable de entrada manual. NOTA:// Usualmente al copiar la ruta esta al pegarla sale entre comillas. De ser el caso estas deben ser omitidas dentro de la ruta eviada. 

Es de aclarar, para la ejecución del proyecto NO ES NECESARIO ejecutar antes el archivo **data_loader.py**, esto se realiza con el fin de determinar que el proceso de lectura y depuración de los datos sea exitoso, mas se realiza de forma automática en este script llamando justamente las funciones empleadas en **data_loader.py**.

Luego de suministrada la ruta, el sistema pedira determinar si se desea imprimir los diagnosticos de cargue y depuración de datos exitosos. De querer ver este disgnóstico se debe digitar **True**, caso contrario digitar **False**.

Finalmente, el sistema pedirá determinar si se desea imprimir el gráfico de entrenamiento del modelo. De querer ver el gráfico se debe digitar **True**, caso contrario digitar **False**.

Una vez ejecutado el proceso el programa realizará de forma automática el cargue, depuración y segmentación de los datos, así como el entrenamiento y exportación del modelo a la carpeta **models**. Tambien guardará los datos de test para su uso en la evaluación.  

4) [evaluate.py](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/6cd9ff0a413aca8cf5e9d284076aa98e2ca083c9/src/evaluate.py)


5) [predict.py](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/6cd9ff0a413aca8cf5e9d284076aa98e2ca083c9/src/predict.py)


6) [utils.py](https://github.com/jeiber-ucentral/Modelo_inmoviliario/blob/6cd9ff0a413aca8cf5e9d284076aa98e2ca083c9/src/utils.py)

En este proceso no se realizaron funciones adicionales, por tanto y hasta el momento no existe nada en este archivo. Por tanto no es necesaria su ejecución. 


## 7. Rendimiento del modelo



## Conclusión

A lo largo de este proyecto


























