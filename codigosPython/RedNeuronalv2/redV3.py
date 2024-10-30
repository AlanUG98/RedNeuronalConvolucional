import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Ruta al directorio que contiene las carpetas de cada clase
data_directory = r'C:\Users\xatla\Desktop\TTv2\datasetV3'


# Clases o etiquetas correspondientes a cada gesto
class_names = ['izquierda', 'avanzar', 'derecha', 'reposo']

# Inicializar ImageDataGenerator para el aumento de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Usar 20% de los datos para validación
)

# Cargar y aumentar imágenes de forma automática desde directorios
train_generator = train_datagen.flow_from_directory(
    data_directory,
    target_size=(100, 100),
    batch_size=32,
    color_mode='grayscale',  # Convertir imágenes a escala de grises
    class_mode='categorical',
    subset='training'  # Parte del conjunto de datos para entrenamiento
)

validation_generator = train_datagen.flow_from_directory(
    data_directory,
    target_size=(100, 100),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation'  # Parte del conjunto de datos para validación
)

# Crear el modelo
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Agregar capa de Dropout para reducir el sobreajuste
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Resumen del modelo
model.summary()

# Callback de EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stopping]
)

# Guardar el modelo entrenado
model.save(r'C:\Users\xatla\Desktop\TTv2\codigosPython\redNeuronalEntrenada\modeloRed3V6.h5')
