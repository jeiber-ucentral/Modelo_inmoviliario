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
Librerías usadas: pandas, numpy, matplotlib, seaborn, etc.
Método de carga: pd.read_csv() para cargar un archivo CSV o funciones similares según el formato del dataset.

### Inspección Inicial
Se realizaron las siguientes acciones para obtener una visión preliminar del conjunto de datos:
Visualización de las primeras filas con df.head().
Verificación de la estructura del dataset con df.info(), que muestra el número de registros, tipos de datos y valores nulos.
Análisis de las estadísticas descriptivas con df.describe(), para obtener un resumen numérico de las variables numéricas.

### Limpieza de Datos
Se identificaron y gestionaron posibles problemas con los datos:
Valores nulos: Se identificaron columnas con valores faltantes y se decidió cómo manejarlos (rellenar, eliminar o imputar).
Valores atípicos: Se identificaron posibles outliers que pudieran afectar los modelos y se decidió si eliminarlos o tratarlos.
Duplicados: Se verificó la presencia de registros duplicados y se eliminaron si fue necesario.

### Análisis Exploratorio de Datos (EDA)
Se realizaron análisis más profundos para entender mejor las relaciones entre las variables:
Distribución de variables: Se analizaron las distribuciones de las variables numéricas y categóricas a través de gráficos como histogramas, diagramas de cajas (boxplots), y gráficos de barras.
Correlaciones: Se calculó la matriz de correlación para identificar posibles relaciones entre variables numéricas.
Relaciones entre variables: Se utilizaron gráficos como diagramas de dispersión (scatter plots) o gráficos de violín (violin plots) para observar las relaciones entre las variables.

### Visualización de Datos
Se utilizaron diversas visualizaciones para presentar los patrones encontrados:
Gráficos de barras, histogramas y diagramas de dispersión.
Gráficos de calor (heatmaps) para visualizar las correlaciones entre las variables.
Diagramas de caja para mostrar la variabilidad y la presencia de valores atípicos.

### Conclusiones Iniciales
A partir de la exploración de los datos, se extrajeron las siguientes conclusiones clave:
[Incluir patrones relevantes que se hayan observado].
[Mencionar posibles problemas de calidad de los datos, como valores nulos, duplicados, etc.].
[Destacar cualquier hallazgo importante o interesante que pueda influir en el análisis posterior].

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


























