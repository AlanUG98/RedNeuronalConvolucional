import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras

# Inicializar Mediapipe para la detección de manos
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Cargar el modelo entrenado
model_path = r'C:\Users\xatla\Desktop\TTv2\codigosPython\redNeuronalEntrenada\modeloRed3V1.h5'
model = keras.models.load_model(model_path)

# Inicializar la cámara web
cap = cv2.VideoCapture(0)
img_width, img_height = 100, 100  # Dimensiones deseadas para la imagen de la mano

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Preprocesamiento para la detección de manos
        copia = image.copy()
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Convertir de vuelta a BGR para mostrar con OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extraer la región de interés de la mano
                all_x, all_y = [], []
                for hnd in mp_hands.HandLandmark:
                    all_x.append(int(hand_landmarks.landmark[hnd].x * image.shape[1]))
                    all_y.append(int(hand_landmarks.landmark[hnd].y * image.shape[0]))

                # Crear un cuadro delimitador cuadrado alrededor de la mano
                center_x, center_y = (min(all_x) + max(all_x)) // 2, (min(all_y) + max(all_y)) // 2
                half_size = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) // 2
                square_size = max(half_size * 2, img_width, img_height)
                min_x, max_x = center_x - square_size // 2, center_x + square_size // 2
                min_y, max_y = center_y - square_size // 2, center_y + square_size // 2

                # Ajustar los bordes del cuadro delimitador para que no estén fuera de la imagen
                min_x, max_x = max(0, min_x), min(image.shape[1], max_x)
                min_y, max_y = max(0, min_y), min(image.shape[0], max_y)

                # Recortar y redimensionar la imagen a 100x100
                cropped_image = copia[min_y:max_y, min_x:max_x]
                if cropped_image.size != 0:
                    cropped_image = cv2.resize(cropped_image, (img_width, img_height))
                    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    cropped_image = cropped_image / 255.0
                    cropped_image = np.expand_dims(cropped_image, axis=[0, -1])

                    # Realizar la predicción
                    prediction = model.predict(cropped_image)
                    gesture_index = np.argmax(prediction)
                    gesture_name = ['gesto1', 'gesto2', 'gesto3'][gesture_index]

                    # Imprimir el gesto detectado
                    print(f"Gesto detectado: {gesture_name}")

        # Mostrar la imagen capturada
        cv2.imshow('Video', image)

        # Romper el bucle si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
