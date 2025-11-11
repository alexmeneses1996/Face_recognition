"""
Sistema de Reconocimiento Facial usando face_recognition y OpenCV
Detecta y reconoce caras en tiempo real desde la webcam o imágenes

Instalación requerida:
pip install face-recognition opencv-python numpy pillow
"""

import face_recognition
import cv2
import numpy as np
import os
from pathlib import Path

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        
    def load_known_faces(self, faces_folder="known_faces"):
        """
        Carga las caras conocidas desde una carpeta
        Estructura: known_faces/nombre_persona/foto.jpg
        """
        faces_path = Path(faces_folder)
        if not faces_path.exists():
            print(f"Creando carpeta {faces_folder}...")
            faces_path.mkdir(parents=True)
            print(f"Por favor, agrega fotos en {faces_folder}/nombre_persona/")
            return
        
        for person_folder in faces_path.iterdir():
            if person_folder.is_dir():
                person_name = person_folder.name
                print(f"Cargando imágenes de {person_name}...")
                
                for image_path in person_folder.glob("*"):
                    if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        try:
                            image = face_recognition.load_image_file(str(image_path))
                            encodings = face_recognition.face_encodings(image)
                            
                            if encodings:
                                self.known_face_encodings.append(encodings[0])
                                self.known_face_names.append(person_name)
                                print(f"  ✓ Cargada: {image_path.name}")
                            else:
                                print(f"  ✗ No se detectó cara en: {image_path.name}")
                        except Exception as e:
                            print(f"  ✗ Error al cargar {image_path.name}: {e}")
        
        print(f"\nTotal de caras cargadas: {len(self.known_face_encodings)}")
    
    def recognize_from_webcam(self):
        """Reconocimiento en tiempo real desde la webcam"""
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("Error: No se puede acceder a la webcam")
            return
        
        print("\nIniciando reconocimiento facial...")
        print("Presiona 'q' para salir")
        
        process_frame = True
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Procesar cada segundo frame para mejorar rendimiento
            if process_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Detectar caras y encodings
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, face_encoding, tolerance=0.6
                    )
                    name = "Desconocido"
                    
                    # Usar la cara con menor distancia
                    if self.known_face_encodings:
                        face_distances = face_recognition.face_distance(
                            self.known_face_encodings, face_encoding
                        )
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                    
                    face_names.append(name)
            
            process_frame = not process_frame
            
            # Dibujar resultados
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Escalar de vuelta al tamaño original
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Dibujar rectángulo
                color = (0, 255, 0) if name != "Desconocido" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Etiqueta con nombre
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(
                    frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1
                )
            
            cv2.imshow('Reconocimiento Facial', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
    
    def recognize_from_image(self, image_path):
        """Reconoce caras en una imagen específica"""
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        # Convertir a BGR para OpenCV
        image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=0.6
            )
            name = "Desconocido"
            
            if self.known_face_encodings:
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    confidence = (1 - face_distances[best_match_index]) * 100
                    name = f"{name} ({confidence:.1f}%)"
            
            # Dibujar resultados
            color = (0, 255, 0) if "Desconocido" not in name else (0, 0, 255)
            cv2.rectangle(image_cv, (left, top), (right, bottom), color, 2)
            cv2.rectangle(image_cv, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(
                image_cv, name, (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1
            )
        
        cv2.imshow('Resultado', image_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    # Crear sistema de reconocimiento
    system = FaceRecognitionSystem()
    
    # Cargar caras conocidas
    system.load_known_faces("known_faces")
    
    if not system.known_face_encodings:
        print("\n⚠️  No se cargaron caras conocidas.")
        print("Crea una carpeta 'known_faces' con subcarpetas para cada persona")
        print("Ejemplo: known_faces/juan/foto1.jpg")
        return
    
    # Menú
    print("\n=== Sistema de Reconocimiento Facial ===")
    print("1. Reconocer desde webcam")
    print("2. Reconocer desde imagen")
    print("3. Salir")
    
    choice = input("\nElige una opción: ")
    
    if choice == "1":
        system.recognize_from_webcam()
    elif choice == "2":
        image_path = input("Ruta de la imagen: ")
        if os.path.exists(image_path):
            system.recognize_from_image(image_path)
        else:
            print("Imagen no encontrada")
    else:
        print("Saliendo...")


if __name__ == "__main__":
    main()