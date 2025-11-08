import cv2

# Cargar el modelo pre-entrenado (detección de caras)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Abrir cámara (0 = cámara del PC)
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()

    # Convertir a escala de grises (mejora la detección)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar caras
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Dibujar recuadros sobre las caras
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Deteccion de Rostros", frame)

    # Salir con tecla ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()


