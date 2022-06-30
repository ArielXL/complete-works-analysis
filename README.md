### Trabajo de Reconocimiento de Estilos Literarios y Evaluación de Similitud con respecto a Escritos Martianos.

El trabajo se propone predecir a partir un texto de entrada, su similitud, en cuanto al estilo, con el estilo empleado por Martí. Para ello se entrena con una serie de documentos escritos por José Martí y con documentos llevados a cabo por otros escritores. La evaluación de similitud se logra con el uso de algoritmos de Inteligencia Artificial, tanto de aprendizaje supervisado (`Naive Bayes`) como de no supervisado (`KMEANS`).

#### Requerimientos para ejecutar el código

Para poder hacer uso del código, se requieren de varias dependencias. En `requirements.txt` se encuentra, para instalarlas basta con ubicarse en la ruta de la carpeta y correr el comando:

```
    pip install -r requirements.txt
```


Para correr el programa, ubicarse en la ruta de la carpeta y escribir el comando:
````
    python3 main.py
````



Para construir los vectores asociados a los datos de entrenamiento, ubicarse en la ruta de la carpeta y correr el comando:

```
    python3 docProcess.py    
```



#### Acerca del programa:

Una vez que se haya ejecutado el comando para correr el programa, se mostrarán en consola las formas de recibir los documentos a evaluar:

1- Hacer uso de los documentos de test ubicados en la carpeta `testData`. Si se desea incorporar algún otro documento se recomienda, que si se conoce de antemano el autor, se ubique en la carpeta correspondiente (si es de Martí en la carpeta `testData/marti` de lo contrario en `testData/otros`)  para poder así evaluar el algoritmo. En caso que no se conozca el escritor ubicar en cualquiera de las dos carpetas y basar el resultado en la predicción obtenida.

2- Introducir el nombre de la ruta (respectiva a la carpeta del proyecto) donde se encuentran los textos que se desean procesar.



Se mostrarán luego los posibles algoritmos a emplear para llevar a cabo esta estimación:

​	1- `Naive Bayes` (Supervisado).

​	2- `Kmeans` (No Supervisado).

​	3- Ambos. En caso de que se seleccione esta opción y los datos a estimar sean los ubicado en la carpeta `testData`, se lleva a cabo una comparación de ambos algoritmos basados en la cantidad de aciertos que se obtuvieron en la predicción en correspondencia a la carpeta en que estaban ubicados (o sea si están en la carpeta `testData/marti` se asume que es de  Martí y análogamente para los ubicados en la carpeta`testData/otros`). 

Se proveen también algunas métricas que evalúan los algoritmos.

En caso que se deseen construir los vectores asociados a los datos de entrenamiento, correr el comando anteriormente dicho. Este algoritmo procesa documentos sumamente extensos, el proceso es sumamente lento. En la carpeta `vector` se encuentran ya ubicados los vectores asociados a los documentos de entrenamiento (los ubicados en `data`), en set.csv.
