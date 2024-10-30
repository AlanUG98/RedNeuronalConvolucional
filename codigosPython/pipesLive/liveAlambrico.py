import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
import serial

rotarImg = True

# Inicializar Mediapipe para la detección de manos
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Nombres de las clases con las que fue entrenado el modelo
class_names = ['avanzar', 'derecha', 'izquierda', 'reposo']

# Cargar el modelo entrenado
model_path = r'C:\Users\xatla\Desktop\TTv2\codigosPython\redNeuronalEntrenada\modeloRed3V1.h5.h5'
model = keras.models.load_model(model_path)

# Inicializar la cámara web
cap = cv2.VideoCapture(0)
img_width, img_height = 100, 100  # Desired image dimensions
ser = serial.Serial('COM3', 9600)


with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        #la sig linea voltea la camara, el pulgar siempre debe de apuntar hacia arriba, borrar la sig linea si el pulgar apunta abajo
        #image = cv2.flip(image, -1)  # El segundo argumento -1 indica un flip vertical y horizontal

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate the bounding box of the hand
                all_x, all_y = [], []
                for hnd in mp_hands.HandLandmark:
                    all_x.append(hand_landmarks.landmark[hnd].x)
                    all_y.append(hand_landmarks.landmark[hnd].y)
                min_x, max_x = int(min(all_x) * image.shape[1]), int(max(all_x) * image.shape[1])
                min_y, max_y = int(min(all_y) * image.shape[0]), int(max(all_y) * image.shape[0])

                # Crop the image to the bounding box of the hand
                # Ensure the bounding box is at least 100x100 and centered around the hand
                bbox_width, bbox_height = max_x - min_x, max_y - min_y
                if bbox_width > img_width or bbox_height > img_height:
                    width_padding = (bbox_width - img_width) // 2 if bbox_width > img_width else 0
                    height_padding = (bbox_height - img_height) // 2 if bbox_height > img_height else 0
                    min_x -= width_padding
                    max_x += width_padding
                    min_y -= height_padding
                    max_y += height_padding
                else:
                    center_x, center_y = (min_x + max_x) // 2, (min_y + max_y) // 2
                    min_x = center_x - img_width // 2
                    max_x = center_x + img_width // 2
                    min_y = center_y - img_height // 2
                    max_y = center_y + img_height // 2

                # Ensure the bounding box does not go outside the image frame
                min_x, min_y = max(min_x, 0), max(min_y, 0)
                max_x, max_y = min(max_x, image.shape[1]), min(max_y, image.shape[0])

                # Crop and resize the image
                cropped_image = image[min_y:max_y, min_x:max_x]
                cropped_image = cv2.resize(cropped_image, (img_width, img_height))
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                cropped_image = cropped_image / 255.0  # Normalization
                cropped_image = np.expand_dims(cropped_image, axis=-1)
                cropped_image = np.expand_dims(cropped_image, axis=0)

                # Predict gesture
                prediction = model.predict(cropped_image)
                class_id = np.argmax(prediction)
                class_name = class_names[class_id]
                ser.write((class_name + '\n').encode())

                # Draw the bounding box and class name
                cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
                cv2.putText(image, class_name, (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                )

        cv2.imshow('Hand Gesture Recognition', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
