import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
import random
import os

# Carga el modelo de red neuronal
modelo = load_model('C:\\Users\\xatla\\Desktop\\TTv2\\codigosPython\\redNeuronalEntrenada\\modeloV4.h5')

# Nombres de las clases con las que fue entrenado el modelo
class_names = ['izquierda', 'avanzar', 'derecha', 'reposo']


def seleccionar_imagenes_aleatorias(carpeta, numero_imagenes):
    archivos = os.listdir(carpeta)
    return random.sample(archivos, min(numero_imagenes, len(archivos)))


def procesar_y_predecir(imagenes, carpeta):
    resultados = []
    for imagen in imagenes:
        # Carga y procesa la imagen
        img = load_img(os.path.join(carpeta, imagen), color_mode='grayscale', target_size=(100, 100))
        img_array = img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Modelo espera una batch dimension

        # Realiza la predicción
        prediccion = modelo.predict(img_array)
        clase_predicha = class_names[np.argmax(prediccion)]

        # Almacena los resultados
        resultados.append((imagen, clase_predicha))
    return resultados


# Carpetas donde están las imágenes para predecir
carpetas = [
    'C:\\Users\\xatla\\Desktop\\TTv2\\datasetV3\\izquierda',
    'C:\\Users\\xatla\\Desktop\\TTv2\\datasetV3\\derecha',
    'C:\\Users\\xatla\\Desktop\\TTv2\\datasetV3\\avanzar',
    'C:\\Users\\xatla\\Desktop\\TTv2\\datasetV3\\reposo'
]

# Procesar y predecir
numero_imagenes_por_carpeta = 40  # Ajusta este número si es necesario
for carpeta in carpetas:
    imagenes_aleatorias = seleccionar_imagenes_aleatorias(carpeta, numero_imagenes_por_carpeta)
    predicciones = procesar_y_predecir(imagenes_aleatorias, carpeta)

    # Imprimir predicciones interpretadas
    for nombre_imagen, clase_predicha in predicciones:
        print(f"Imagen: {nombre_imagen} - Clase Predicha: {clase_predicha}")

