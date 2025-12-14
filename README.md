Trabajo Practico N° 3 para la materia Procesamiento de Imagenes

En el siguiente Repositorio "TP3_PDI_2025" se presentan la solución del Trabajo Practico N° 3 de Procesamiento de Imagenes.

El archivo .py fue pensado para que pueda ser ejecutado en one-shot

Para ejecutar el mismo, debe seguir los siguientes pasos:

Clonar el repositorio con el siguiente comando en Consola --> gh repo clone jgottig/TP3_PDI_2025 Previamente, debe tener instaladas las librerías necesarias con los siguientes comandos: 
pip install opencv-python 
pip install numpy  
Ademas, utilizamos librerías nativas de Python

Para su ejecución, Abrir su IDE Favorito. En algunos casos recomendamos que utilice la versión de Python 3.10 (Dado que OpenCV puede tener fallas en Versiones 3.11 +)

Consideraciones para la ejecución:
Para su ejecución, solo debe dar "Play" al archivo "tiradas.py" en su IDE, RUN, o bien ejecutar el código principal por consola. A medida que el codigo se va ejecutando, 
Se irán reproduciendo los videos en una ventana, y a su vez, en back, se estarán procesando las imagenes para detectar los dados y sus valores.
Aguarde a que finalicen de reproducirse los 4 videos automaticamente y luego revise la carpeta "tiradas_salida" que se creó dentro del mismo proyecto.

Estructura: 
Archivo tiradas.py con el código principal a ejecutar
Carpeta "frames" que contiene los videos con las tiradas de dados, todas las tiradas deben respetar la estructura del nombre "tirada_*.mp4"
Carpeta "tiradas_salida", la misma será creada automaticamente luego de completarse la ejecución del código, y almacenara los videos de salida de cada tirada

Analice los resultados en la consola de salida.

En caso de querer también visualizar los videos con la aplicación de las mascaras, puede descomentar la linea #14 " # ver_mas = "QUIERO VER EL PASO A PASO" ", de esta forma, podrá ver 3 videos en paralelo.
