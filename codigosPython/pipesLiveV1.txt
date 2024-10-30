import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Cargar el modelo
loaded_model = load_model(r'C:\Users\xatla\Desktop\TTv2\codigosPython\redNeuronalEntrenada\modelo_gestos.h5')

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Definir región de interés (ROI)
    roi = frame[100:400, 320:620]
    cv2.imshow('roi', roi)

    # Preprocesar la ROI para que coincida con el formato del modelo
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(roi_gray, (100, 100), interpolation=cv2.INTER_AREA)
    roi_normalized = roi_resized / 255.0
    roi_input = roi_normalized.reshape(1, 100, 100, 1)

    # Realizar la predicción con el modelo cargado

    predictions = loaded_model.predict(roi_input, verbose=0)
    prediction = np.argmax(predictions[0])

    # Mostrar el resultado en el cuadro de video
    copy = frame.copy()
    cv2.rectangle(copy, (320, 100), (620, 400), (250, 0, 0), 5)
    cv2.putText(copy, f'Gesture: {prediction}', (300, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', copy)

    # Salir del bucle si se presiona la tecla 'Enter'
    if cv2.waitKey(1) == 13:
        break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()


