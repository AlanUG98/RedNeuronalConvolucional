from tensorflow.keras.models import load_model
import cv2
import mediapipe as mp
import numpy as np


"""def getLetter(result):
    classLabels = {0: 'H',
                   1: 'O',
                   2: 'L',
                   3: 'A'
                   }
    try:

        print(result)
        x = list(result)
        print(x)
        print(x[0])
        print(x[1])
        print(x[2])
        res = x.index(1)
        return classLabels[res]
    except:
        return ""
        """

def getGesture(result):
    classLabels = {0: 'gesto1',
                  1: 'gesto2',
                  2: 'gesto3'}
    try:
        # Convierte las probabilidades a una lista y encuentra el índice del valor más alto
        max_index = np.argmax(result)
        # Devuelve la etiqueta correspondiente al índice encontrado
        return classLabels[max_index]
    except:
        return ""


#classifier = load_model("/Users/miguelsanchezbrito/Desktop/Evaluacion/my_gestures_cnn.h5")
classifier = load_model(r'C:\Users\xatla\Desktop\TTv2\codigosPython\redNeuronalEntrenada\modelo_gestos.h5')
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        copia = image.copy()
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                all_x, all_y = [], []
                for hnd in mp_hands.HandLandmark:
                    all_x.append(int(hand_landmarks.landmark[hnd].x * image.shape[1]))
                    all_y.append(int(hand_landmarks.landmark[hnd].y * image.shape[0]))
                start_point = (min(all_x), min(all_y))
                end_point = (max(all_x), max(all_y))
                color = (0, 255, 0)
                thickness = 2
                image = cv2.rectangle(image, start_point, end_point, color, thickness)
                cropped_image = copia[min(all_y):max(all_y), min(all_x):max(all_x)]
                new_size = (100, 100)
                cropped_image = cv2.resize(cropped_image, new_size)

                # cv2.imshow(' corte', cv2.flip(cropped_image, 1))
                roi = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                roi = roi.reshape(1, 100, 100, 1)
                roi = roi / 255
                result = classifier.predict(roi, 1, verbose=0)[0]
                detected_gesture = getGesture(result)
                print(f"El gesto identificado es: {detected_gesture}")
                cv2.putText(copia, detected_gesture, (50, 150), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)

        cv2.imshow(' Manos', copia)


        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()

print("Hola mundo")