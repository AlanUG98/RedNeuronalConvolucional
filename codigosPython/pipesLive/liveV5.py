import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras

# Inicializar Mediapipe para la detección de manos
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Cargar el modelo entrenado
model_path = r'C:\Users\xatla\Desktop\TTv2\codigosPython\redNeuronalEntrenada\modeloRed3V1.h5'
model = keras.models.load_model(model_path)

# Inicializar la cámara web
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Preprocesamiento para la detección de manos
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Convertir de vuelta a BGR para mostrar con OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Si se detectan manos, dibujar las marcas y extraer la imagen
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar las marcas de la mano
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                )

            # Extrae la imagen con las marcas de la mano detectada
            marked_image = image.copy()

            # Realizar la predicción y demás procesos aquí...
            # Tu código de clasificación y procesamiento adicional iría aquí
            # Asumiendo que se procesa la primera mano detectada para la clasificación
            hand_landmarks = results.multi_hand_landmarks[0]
            # Procesamiento de la imagen para clasificación
            # ... (por ejemplo, extracción de la región de la mano, redimensionamiento, etc.)
            # Predicción con el modelo de red neuronal
            # ... (código para la predicción)
            # Recuerda que necesitas procesar la imagen de la misma manera que hiciste durante el entrenamiento

            # Aquí va el procesamiento de la imagen para la clasificación
            # (Este es un ejemplo genérico basado en el código anterior)
            # Deberías ajustar esto para que coincida exactamente con tu preprocesamiento de entrenamiento
            cropped_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)  # Convertir a escala de grises
            cropped_image = cv2.resize(cropped_image, (100, 100))  # Redimensionar a 100x100
            cropped_image = cropped_image / 255.0  # Normalizar
            cropped_image = np.expand_dims(cropped_image, axis=-1)  # Añadir dimensión de canal
            cropped_image = np.expand_dims(cropped_image, axis=0)  # Añadir dimensión de batch

            # Realizar la predicción
            prediction = model.predict(cropped_image)
            gesture_index = np.argmax(prediction)
            gesture_name = ['gesto Derecha', 'gesto Izquierda', 'gesto Reposo', 'gesto Avanzar'][gesture_index]

            # Imprimir el gesto detectado
            print(f"Gesto detectado: {gesture_name}")

        else:
            print("No se detecta mano")

        # Mostrar la imagen capturada
        cv2.imshow('Manos detectadas', image)
        if results.multi_hand_landmarks:
            cv2.imshow('Mano marcada', marked_image)

        # Romper el bucle si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
