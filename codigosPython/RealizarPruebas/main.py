import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model


# Cargar el modelo entrenado
loaded_model = load_model(r'C:\Users\xatla\Desktop\TTv2\codigosPython')

# Ruta al directorio que contiene las imágenes de prueba
#test_data_directory = r'C:\Users\xatla\Desktop\TTv2\datasetFInal'
test_data_directory = r'C:\Users\xatla\Desktop\TTv2\datasetFInal\pruebasexternas'


# Clases correspondientes a cada gesto
class_names = ['gesto1', 'gesto2', 'gesto3']

# Inicializar listas para almacenar las imágenes y las etiquetas reales
test_images = []
test_labels = []

# Recorrer el directorio de datos de prueba
for class_name in class_names:
    class_directory = os.path.join(test_data_directory, class_name)

    # Leer todas las imágenes de prueba y asignar la etiqueta correspondiente
    for filename in os.listdir(class_directory):
        image = cv2.imread(os.path.join(class_directory, filename), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (100, 100))
        test_images.append(image)
        test_labels.append(class_names.index(class_name))

# Convertir las listas en matrices numpy
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Realizar predicciones en las imágenes de prueba
predictions = loaded_model.predict(test_images)

# Recorrer las predicciones y mostrar etiqueta original y predicha
for i in range(len(predictions)):
    original_label = class_names[test_labels[i]]  # Etiqueta original
    predicted_label = class_names[np.argmax(predictions[i])]  # Etiqueta predicha

    print(f'Imagen {i + 1} - Etiqueta original: {original_label}, Etiqueta predicha: {predicted_label}')
