import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras

# Inicializar Mediapipe para la detección de manos
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Cargar el modelo entrenado
model_path = r'C:\Users\xatla\Desktop\TTv2\codigosPython\redNeuronalEntrenada\modeloV3.h5'
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
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Convertir de vuelta a BGR para mostrar con OpenCV
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extraer la región de interés de la mano
                x_coords = [landmark.x * image.shape[1] for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y * image.shape[0] for landmark in hand_landmarks.landmark]
                bounding_rect = cv2.boundingRect(np.array([(x, y) for x, y in zip(x_coords, y_coords)], dtype=np.int32))

                # Encuentra el cuadrado más grande posible dentro del bounding_rect
                x, y, w, h = bounding_rect
                square_side = max(w, h)
                x_center = x + w / 2
                y_center = y + h / 2
                if square_side > 0:
                    x_start = int(x_center - square_side / 2)
                    y_start = int(y_center - square_side / 2)
                    x_end = int(x_center + square_side / 2)
                    y_end = int(y_center + square_side / 2)

                    # Asegúrate de que el cuadrado no salga de los límites de la imagen
                    x_start, y_start = max(0, x_start), max(0, y_start)
                    x_end, y_end = min(image.shape[1], x_end), min(image.shape[0], y_end)

                    # Recorta la imagen a este cuadrado
                    cropped_image = image[y_start:y_end, x_start:x_end]

                    # Redimensiona la imagen recortada a las dimensiones deseadas y la convierte a escala de grises
                    cropped_image = cv2.resize(cropped_image, (img_width, img_height))
                    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

                    # Normaliza la imagen
                    cropped_image = cropped_image / 255.0

                    # Añade una dimensión de canal
                    cropped_image = np.expand_dims(cropped_image, axis=-1)
                    cropped_image = np.expand_dims(cropped_image, axis=0)

                    # Realizar la predicción
                    prediction = model.predict(cropped_image)
                    gesture_index = np.argmax(prediction)
                    gesture_name = ['gestoDerecha', 'gestoIzquierda', 'gestoReposo'][gesture_index]

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
