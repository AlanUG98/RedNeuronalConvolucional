import os
import numpy as np
from tensorflow import keras
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Ruta al directorio que contiene las carpetas "gesto1", "gesto2" y "gesto3"
data_directory = r'C:\Users\xatla\Desktop\TTv2\datasetV3'


# Clases o etiquetas correspondientes a cada gesto
class_names = ['izquierda', 'avanzar', 'derecha', 'reposo']

# Inicializar listas para almacenar imágenes y etiquetas
images = []
labels = []

# Recorre cada clase
for class_name in class_names:
    class_directory = os.path.join(data_directory, class_name)

    # Lee todas las imágenes de la clase y asigna la etiqueta correspondiente
    for filename in os.listdir(class_directory):
        image = cv2.imread(os.path.join(class_directory, filename), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (100, 100))  # Redimensiona a 100x100 si es necesario
        images.append(image)
        labels.append(class_names.index(class_name))

# Convierte las listas en matrices numpy
images = np.array(images)
labels = np.array(labels)

# Divide el conjunto de datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convertir las etiquetas a formato one-hot
y_train_one_hot = to_categorical(y_train, num_classes=4)  # 3 es el número de clases
y_test_one_hot = to_categorical(y_test, num_classes=4)

#-----------------------------------------------------------------

# Crear un modelo de CNN
model = models.Sequential()

# Capas de convolución y max-pooling
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))


# Aplanar la salida de las capas anteriores
model.add(layers.Flatten())

# Capas completamente conectadas
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  # Agregar dropout para reducir el sobreajuste
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))  # 3 clases para los 3 gestos

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Resumen del modelo
model.summary()

# Entrenar el modelo (utiliza tus datos de entrenamiento)
#model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Entrenar el modelo utilizando las etiquetas en formato one-hot
model.fit(X_train, y_train_one_hot, epochs=20, batch_size=32, validation_split=0.2)

# Guardar el modelo entrenado en la carpeta especificada
model.save(r'C:\Users\xatla\Desktop\TTv2\codigosPython\redNeuronalEntrenada\modeloV4.h5')

#--------------------------PRUEBAS---------------------------------
"""
# Realizar predicciones en el conjunto de prueba
predictions = model.predict(X_test)

# Clases correspondientes a cada gesto
class_names = ['gesto1', 'gesto2', 'gesto3']

# Recorrer las predicciones y mostrar etiqueta original y predicha
for i in range(len(predictions)):
    original_label = class_names[y_test[i]]  # Etiqueta original
predicted_label = class_names[np.argmax(predictions[i])]  # Etiqueta predicha

print(f'Imagen {i + 1} - Etiqueta original: {original_label}, Etiqueta predicha: {predicted_label}')

"""
