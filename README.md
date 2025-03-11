# Modelo_inmoviliario
Taller DL de predicción del precio inmoviliario usando Backpropagation

Este proyecto desarrolla un modelo de red neuronal backpropagation para predecir el precio de alquiler de viviendas en una ciudad. Se estructura un repositorio incluyendo codigo modular, documentacion clara y ejemplos reproducibles.

Voy a dividir el análisis en los siguientes apartados:

1. Comprender el problema
2. Estudio univariable
3. Estudio multivariable
4. Limpieza básica de los datos
5. Comprobación de suposiciones

## 1. Objetivo y Datos Utilizados

Dado un conjunto de datos con informacion sobre alquileres (metros cuadrados, numero de habitaciones, ubicacion, antiguedad del inmueble, etc.), se requiere predecir el precio mensual. Esta es una problematica real en aplicaciones inmobiliarias y de analisis de mercado.

Este taller utilizara un conjunto de datos que contiene informacion detallada sobre mas de 4700 propiedades residenciales disponibles para alquiler, abarcando casas, apartamentos y pisos. El conjunto de datos incluye las siguientes variables. 

- BHK: N ́umero de habitaciones, sala y cocina.
- Rent: Precio de alquiler de las casas/apartamentos/pisos.
- Size: Tama ̃no de las casas/apartamentos/pisos en pies cuadrados.
- Floor: Piso en el que se encuentra la propiedad y el total de pisos en el edificio (Ejemplo: Planta baja de 2, 3 de 5, etc.).
- Area Type: C ́alculo del tama ̃no en Superficie Total, Superficie  ́util o Area Construida.  ́
- Area Locality: Localidad de la propiedad.
- City: Ciudad donde est ́a ubicada la propiedad.
- Furnishing Status: Estado de amueblado de la propiedad (Amueblado, Semi-Amueblado o No Amueblado).
- Tenant Preferred: Tipo de inquilino preferido por el due ̃no o agente.
- Bathroom: N ́umero de banos.
- Point of Contact: Persona de contacto para obtener mas informacion sobre la propiedad.

## 2. Explorción del DataSet

La variable 'SalePrice' es la variable objetivo de este conjunto de datos. En pasos posteriores a este análisis exploratorio de datos se realizaría una predicción del valor de esta variable, por lo que voy a estudiarla con mayor detenimiento. A simple vista se pueden apreciar:

* Una desviación con respecto a la distribución normal.
* Una asimetría positiva.
* Algunos picos.

## 3. Modelo de Red Neuronal

### Requerimientos en librerías

* 'GrLivArea' y 'TotalBsmtSF' mantienen una relación lineal positiva con 'SalePrice', aumentando en el mismo sentido. En el caso de 'TotalBsmtSF', la pendiente de esta relación es muy acentuada.
* 'OverallQual' y 'YearBuilt' también parecen relacionadas con 'SalePrice' (más fuerte en el primer caso), tal y como se puede observar en los diagramas de cajas.

Sólo he explorado cuatro variables, pero hay muchas otras a analizar.


## 4. Requisitos previos y dependencias

Hasta ahora sólo me he dejado llevar por la intuición para el análisis de las variables que he creído importantes. Es hora de un análisis más objetivo.

Para ello voy a realizar las siguientes pruebas de correlación:
* Matriz de correlación general: El mapa de calor es una forma visual muy útil para para conocer las variables y sus relaciones. A primera vista hay dos variables que llaman la atención: 'TotalBsmtSF' y '1stFlrSF', seguidas por las variables 'Garage*X*'. En ambos casos parece haber una correlación significativa; en realidad es tan fuerte que podría indicar multicolinealidad, es decir, que básicamente ofrecen la misma información. Con respecto a las correlaciones de la variable 'SalePrice', destacan las vistas anteriormente ('GrLivArea', 'TotalBsmtSF' y 'OverallQual'), pero hay otras que también deberían ser tenidas en cuenta.
* Matriz de correlación centrada en la variable 'SalePrice': En estas matrices de correlación se puede observar:
  * 'OverallQual', 'GrLivArea' y 'TotalBsmtSF' están fuertemente correladas con 'SalePrice'.
  * 'GarageCars' y 'GarageArea' también están fuertemente correladas pero, como he comentado anteriormente, el número de coches que se pueden aparcar en un garaje es una consecuencia de su superficie. Es por esto que sólo voy a mantener una de estas variables en el análisis, 'GarageCars', ya que está más correlada con 'SalePrice'.
  * 'TotalBsmtSF' y '1stFloor' plantean la misma situación. En este caso mantendré 'TotalBsmtSF'.
  * 'FullBath' también está correlada con 'SalePrice'. Parece que a la gente le gusta darse un baño en casa...
  * 'TotRmsAbvGrd' y 'GrLivArea', otro caso de multicolinealidad.
  * 'YearBuilt' también está ligeramente correlada con 'SalePrice'. 
* Diagramas de dispersión entre las variables más correladas.


## 5. Entrenamiento desde train.py

### Datos desaparecidos

Antes de tratar los datos faltantes o missing values, es importante determinar su prevalencia y su aleatoriedad, ya que pueden implicar una reducción del tamaño de la muestra. También hay que asegurarse que la gestión de los datos desaparecidos no esté sesgada o esconda una verdad incómoda.

Por razones prácticas voy a eliminar las variables con más de un 15% de datos faltantes (p.ej. 'PoolQC', 'MiscFeature', 'Alley', etc.); no creo que las echemos de menos, no parecen aspectos importantes a considerar al comprar una casa.

Con respecto a las variables 'Garage*X*', observo el mismo número de datos desaparecidos, hecho que quizás habría que estudiar con más detenimiento. Pero, dado que la información más relevante en cuanto al garaje ya está recogida por la variable 'GarageCars', y que sólo se trata de un 5% de datos faltantes, borraré las citadas variables 'Garage*X*', además de las 'Bsmt*X*' bajo la misma lógica.

En cuanto a las variables 'MasVnrArea' y 'MasVnrType', se puede decir que no son esenciales y que, incluso, tienen una fuerte correlación con 'YearBuilt' y 'OverallQual'. No parece que se vaya a perder mucha información si elimino 'MasVnrArea' and 'MasVnrType'.

Para finalizar, encuentro un dato faltante en la variable 'Electrical'. Ya que sólo se trata de una observación, voy a borrarla y a mantener la variable.

En resumen, voy a borrar todas las variables con datos desaparecidos, excepto la variable 'Electrical'; en este caso sólo voy a borrar la observación con el dato faltante.

### Datos atípicos

Los datos atípicos u outliers pueden afectar marcadamente el modelo, además de suponer una fuente de información en sí misma. Su tratamiento es un asunto complejo que requiere más atención; por ahora sólo voy a hacer un análisis rápido a través de la desviación estándar de la variable 'SalePrice' y a realizar un par de diagramas de dispersión.

#### Análisis univariable

La primera tarea en este caso es establecer un umbral que defina una observación como valor atípico. Para ello voy a estandarizar los datos, es decir, transformar los valores datos para que tengan una media de 0 y una desviación estándar de 1.

* Los valores bajos son similares y no muy alejados del 0.
* Los valores altos están muy alejados del 0. Los valores superiores a 7 están realmente fuera de rango.

#### Análisis bivariable

Este diagrama de dispersión muestra un par de cosas interesantes:

* Los dos valores más altos de la variable 'GrLivArea' resultan extraños. Sólo puedo especular, pero podría tratarse de terrenos agrícolas o muy degradados, algo que explicaría su bajo precio. Lo que está claro es que estos dos puntos son atípicos, por lo que voy a proceder a eliminarlos.
* Las dos observaciones más altas de la variable 'SalePrice' se corresponden con las que observamos en el análisis univariable anterior. Son casos especiales, pero parece que siguen la tendencia general, por lo que voy a mantenerlas.

Aunque se pueden observar algunos valores bastante extremos (p.ej. TotalBsmtSF > 3000), parece que conservan la tendencia, por lo que voy a mantenerlos.


## 6. Evaluando desde evaluate.py

Ya he realizado cierta limpieza de datos y estudiado la variable 'SalePrice'. Ahora voy a comprobar si 'SalePrice' cumple las asunciones estadísticas que nos permiten aplicar las técnicas del análisis multivariable.

De acuerdo con [Hair et al. (2013)](https://www.amazon.com/gp/product/9332536503/), hay que comprobar cuatro suposiciones fundamentales:

* <b>Normalidad</b> - Cuando hablamos de normalidad lo que queremos decir es que los datos deben parecerse a una distribución normal. Es importante porque varias pruebas estadísticas se basan en esta suposición. Sólo voy a comprobar la normalidad de la variable 'SalePrice', aunque resulte un tanto limitado ya que no asegura la normalidad multivariable. Además, si resolvemos la normalidad evitamos otros problemas, como la homocedasticidad.

* <b>Homocedasticidad</b> - La homocedasticidad se refiere a la suposición de que las variables dependientes tienen el mismo nivel de varianza en todo el rango de las variables predictoras, según [(Hair et al., 2013)](https://www.amazon.com/gp/product/9332536503/). La homocedasticidad es deseable porque queremos que el término de error sea el mismo en todos los valores de las variables independientes.

* <b>Linealidad</b>- La forma más común de evaluar la linealidad es examinar los diagramas de dispersión y buscar patrones lineales. Si los patrones no son lineales, valdría la pena explorar las transformaciones de datos. Sin embargo, no voy a entrar en esto porque la mayoría de los gráficos de dispersión que hemos visto parecen tener relaciones lineales.

* <b>Ausencia de errores correlacionados</b> - Esto ocurre a menudo en series temporales, donde algunos patrones están relacionados en el tiempo. Tampoco voy a tocar este asunto.

## 7. Prediciendo desde predict.py

El objetivo es estudiar la variable 'SalePrice' de forma fácil, comprobando:

* <b>Histograma</b> - Curtosis y asimetría.
* <b>Gráfica de probabilidad normal</b> - La distribución de los datos debe ajustarse a la diagonal que representa la distribución normal.

De estos gráficos se desprende que 'SalePrice' no conforma una distribución normal. Muestra picos, asimetría positiva y no sigue la línea diagonal; aunque una simple transformación de datos puede resolver el problema. 

Terminado el trabajo con 'SalePrice', voy a seguir con 'GrLivArea'. La variable 'GrLivArea' muestra asimetría.

Prosigo con el estudio de la variable 'TotalBsmtSF'. Estos gráficos nos muestran que la variable 'TotalBsmtSF':

* Presenta asimetrías.
* Hay un número significativo de observaciones con valor cero (casas sin sótano).
* El valor cero no nos permite hacer transformaciones logarítmicas.

Para aplicar una transformación logarítmica, crearé una variable binaria (tener o no tener sótano). Después, aplicaré la transformación logarítmica a todas las observaciones que no sean cero, ignorando aquellas con valor cero. De esta manera podré transformar los datos, sin perder el efecto de tener o no sótano.

## 8. Rendimiento del modelo

El mejor método para probar la homocedasticidad para dos variables métricas es de forma gráfica. Las desviaciones de una dispersión uniforme se muestran mediante formas tales como conos (pequeña dispersión a un lado del gráfico, gran dispersión en el lado opuesto) o diamantes (un gran número de puntos en el centro de la distribución).

Empiezo por 'SalePrice' y 'GrLivArea'. Las anteriores versiones de este gráfico de dispersión (antes de las transformaciones logarítmicas), tenían una forma cónica. Como puede apreciarse, el gráfico actual ya no tiene una forma cónica. Tan solo asegurando la normalidad en algunas variables, hemos resuelto el problema de la homocedasticidad.

Ahora vamos a comprobar 'SalePrice' con 'TotalBsmtSF'. Podemos decir que, en general, la variable 'SalePrice' muestra niveles equivalentes de varianza en todo el rango de 'TotalBsmtSF'.

Convierto las variables categóricas en variables ficticias o dummies.


## Conclusión

A lo largo de este kernel he puesto en práctica muchas de las estrategias propuestas por [Hair et al. (2013)](https://www.amazon.com/gp/product/9332536503/). He estudiado las variables, analizado 'SalePrice' a solas y con las variables más correladas, he lidiado con datos faltantes y valores atípicos, he probado algunos de los supuestos estadísticos fundamentales e incluso he transformado variables categoriales en variables dummy. Todo un abanico de técnicas en Python, usando las librerías [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [NumPy](https://www.numpy.org/), [SciPy](https://www.scipy.org/) y [Scikit-learn](https://scikit-learn.org/stable/).

Se han quedado algunos asuntos en el tintero, pero no ha estado mal para empezar.

No quiero finalizar este ejercicio sin dar antes las gracias públicamente a Pedro Marcelino por su magnífico trabajo, del que éste es poco más que una traducción y retoque.


## Referencias
* [Pedro Marcelino](https://www.kaggle.com/pmarcelino)
* [Hair et al., 2013, Multivariate Data Analysis, 7th Edition](https://www.amazon.com/gp/product/9332536503/)

























