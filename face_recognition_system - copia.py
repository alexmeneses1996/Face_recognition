"""
Sistema de Reconocimiento Facial con MediaPipe
Instalación: pip install mediapipe opencv-python

Autor: Sistema optimizado para Windows
Versión: 2.0
"""

import cv2
import mediapipe as mp
import pickle
import os
from pathlib import Path
import numpy as np


class ReconocimientoFacial:
    def __init__(self):
        # Inicializar MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Configurar detección de caras
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0=cercano, 1=lejano
            min_detection_confidence=0.6
        )
        
        # Configurar malla facial para encodings
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        
        self.caras_conocidas = {}
        self.archivo_encodings = "caras_guardadas.pkl"
        
        print("✓ Sistema de reconocimiento facial inicializado")
    
    def extraer_caracteristicas(self, imagen):
        """Extrae características faciales únicas de una imagen"""
        try:
            rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            resultado = self.face_mesh.process(rgb)
            
            if not resultado.multi_face_landmarks:
                return None
            
            # Usar landmarks clave como características
            landmarks = []
            cara = resultado.multi_face_landmarks[0]  # Primera cara
            
            # Extraer TODOS los landmarks (468 puntos) para mejor precisión
            for lm in cara.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            # Normalizar características
            landmarks = np.array(landmarks)
            landmarks = (landmarks - landmarks.mean()) / (landmarks.std() + 1e-6)
            
            return landmarks
        
        except Exception as e:
            print(f"Error al extraer características: {e}")
            return None
    
    def calcular_similitud(self, cara1, cara2):
        """Calcula qué tan parecidas son dos caras (0-1)"""
        if cara1 is None or cara2 is None:
            return 0.0
        
        try:
            # Normalizar longitud
            min_len = min(len(cara1), len(cara2))
            cara1 = cara1[:min_len]
            cara2 = cara2[:min_len]
            
            # Calcular similitud con distancia euclidiana
            distancia = np.linalg.norm(cara1 - cara2)
            # Escala ajustada: distancia pequeña = similitud alta
            similitud = np.exp(-distancia / 2)  # Distribución gaussiana
            
            return float(similitud)
        except:
            return 0.0
    
    def cargar_caras_conocidas(self, carpeta="known_faces"):
        """Carga las fotos de personas conocidas"""
        print(f"\n{'='*50}")
        print("CARGANDO CARAS CONOCIDAS")
        print(f"{'='*50}")
        
        # Intentar cargar desde archivo guardado
        if os.path.exists(self.archivo_encodings):
            try:
                with open(self.archivo_encodings, 'rb') as f:
                    self.caras_conocidas = pickle.load(f)
                
                print(f"✓ Cargadas {len(self.caras_conocidas)} personas desde archivo")
                for nombre in self.caras_conocidas.keys():
                    print(f"  • {nombre}")
                print(f"{'='*50}\n")
                return
            except Exception as e:
                print(f"⚠ Error al cargar archivo: {e}")
                print("Generando nuevos encodings...\n")
        else:
            print("test")
        # Crear carpeta si no existe
        ruta_carpeta = Path(carpeta)
        if not ruta_carpeta.exists():
            ruta_carpeta.mkdir(parents=True)
            print(f"✓ Carpeta '{carpeta}' creada")
            print(f"\n📁 INSTRUCCIONES:")
            print(f"1. Crea subcarpetas con el nombre de cada persona")
            print(f"2. Agrega 2-5 fotos de cada persona en su carpeta")
            print(f"Ejemplo: {carpeta}/juan/foto1.jpg")
            print(f"{'='*50}\n")
            return
        
        # Procesar cada persona
        total_procesadas = 0
        
        for carpeta_persona in ruta_carpeta.iterdir():
            if not carpeta_persona.is_dir():
                continue
            
            nombre = carpeta_persona.name
            print(f"\n👤 Procesando: {nombre}")
            print("-" * 40)
            
            encodings_persona = []
            
            for archivo_foto in carpeta_persona.glob("*"):
                if archivo_foto.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    continue
                
                try:
                    imagen = cv2.imread(str(archivo_foto))
                    
                    if imagen is None:
                        print(f"  ✗ No se pudo leer: {archivo_foto.name}")
                        continue
                    
                    # Redimensionar si es muy grande
                    h, w = imagen.shape[:2]
                    if w > 800:
                        escala = 800 / w
                        imagen = cv2.resize(imagen, None, fx=escala, fy=escala)
                    
                    encoding = self.extraer_caracteristicas(imagen)
                    
                    if encoding is not None:
                        encodings_persona.append(encoding)
                        print(f"  ✓ {archivo_foto.name}")
                        total_procesadas += 1
                    else:
                        print(f"  ✗ Sin cara detectada: {archivo_foto.name}")
                
                except Exception as e:
                    print(f"  ✗ Error con {archivo_foto.name}: {e}")
            
            if encodings_persona:
                self.caras_conocidas[nombre] = encodings_persona
                print(f"  → Total: {len(encodings_persona)} fotos cargadas")
        
        # Guardar encodings
        if self.caras_conocidas:
            try:
                with open(self.archivo_encodings, 'wb') as f:
                    pickle.dump(self.caras_conocidas, f)
                print(f"\n✓ Encodings guardados en '{self.archivo_encodings}'")
            except Exception as e:
                print(f"\n⚠ No se pudo guardar: {e}")
        
        print(f"\n{'='*50}")
        print(f"RESUMEN: {len(self.caras_conocidas)} personas | {total_procesadas} fotos")
        print(f"{'='*50}\n")
    
    def reconocer_cara(self, encoding):
        """Identifica a quién pertenece una cara"""
        if not self.caras_conocidas or encoding is None:
            return "Desconocido", 0.0
        
        mejor_coincidencia = "Desconocido"
        mejor_similitud = 0.0
        
        for nombre, encodings_lista in self.caras_conocidas.items():
            for encoding_conocido in encodings_lista:
                similitud = self.calcular_similitud(encoding, encoding_conocido)

                print(f"Comparando con {nombre}: similitud={similitud:.3f}")
                
                if similitud > mejor_similitud:
                    mejor_similitud = similitud
                    mejor_coincidencia = nombre
        
        # Umbral mínimo de confianza (reducido para mejor detección)
        if mejor_similitud < 0.50:
            return "Desconocido", mejor_similitud
        
        return mejor_coincidencia, mejor_similitud
    
    def webcam_tiempo_real(self):
        """Reconocimiento facial desde la webcam"""
        print("\n" + "="*50)
        print("MODO WEBCAM - Presiona 'q' para salir")
        print("="*50 + "\n")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: No se puede acceder a la webcam")
            print("Verifica que:")
            print("  1. La webcam esté conectada")
            print("  2. No esté siendo usada por otra aplicación")
            return
        
        # Configurar resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✓ Webcam activa\n")
        
        frame_count = 0
        detecciones_cache = []  # Guardar detecciones anteriores
        nombre_cache = "Desconocido"
        confianza_cache = 0.0
        
        with self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        ) as detector:
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠ Error al capturar frame")
                    break
                
                frame_count += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resultados = detector.process(rgb)
                
                # Actualizar detecciones cada frame
                if resultados.detections:
                    detecciones_cache = resultados.detections
                
                # Dibujar TODAS las detecciones (suaviza el titeleo)
                for deteccion in detecciones_cache:
                    bbox = deteccion.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    ancho = int(bbox.width * w)
                    alto = int(bbox.height * h)
                    
                    # Extraer región de la cara
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(w, x + ancho), min(h, y + alto)
                    cara_roi = frame[y1:y2, x1:x2]
                    
                    if cara_roi.size > 0:
                        # Procesar reconocimiento cada 3 frames (optimización)
                        if frame_count % 3 == 0:
                            encoding = self.extraer_caracteristicas(cara_roi)
                            nombre_cache, confianza_cache = self.reconocer_cara(encoding)
                        
                        # Dibujar siempre con último resultado (sin titeleo)
                        color = (0, 255, 0) if nombre_cache != "Desconocido" else (0, 0, 255)
                        cv2.rectangle(frame, (x, y), (x + ancho, y + alto), color, 2)
                        
                        # Etiqueta
                        etiqueta = f"{nombre_cache} ({confianza_cache*100:.0f}%)"
                        tam_texto = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        
                        cv2.rectangle(frame, (x, y - 35), (x + tam_texto[0] + 10, y), color, -1)
                        cv2.putText(frame, etiqueta, (x + 5, y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Instrucciones en pantalla
                cv2.putText(frame, "Presiona 'q' para salir", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Reconocimiento Facial', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Webcam cerrada")
    
    def analizar_imagen(self, ruta_imagen):
        """Analiza una imagen y reconoce las caras"""
        print(f"\nAnalizando: {ruta_imagen}")
        
        if not os.path.exists(ruta_imagen):
            print(f"❌ Error: Archivo no encontrado")
            return
        
        imagen = cv2.imread(ruta_imagen)
        
        if imagen is None:
            print(f"❌ Error: No se puede leer la imagen")
            return
        
        # Redimensionar si es muy grande
        h, w = imagen.shape[:2]
        if w > 1200:
            escala = 1200 / w
            imagen = cv2.resize(imagen, None, fx=escala, fy=escala)
        
        rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        
        with self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.6
        ) as detector:
            
            resultados = detector.process(rgb)
            
            if not resultados.detections:
                print("⚠ No se detectaron caras en la imagen")
                cv2.imshow('Sin caras detectadas', imagen)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return
            
            print(f"✓ {len(resultados.detections)} cara(s) detectada(s)\n")
            
            for i, deteccion in enumerate(resultados.detections, 1):
                bbox = deteccion.location_data.relative_bounding_box
                h, w, _ = imagen.shape
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                ancho = int(bbox.width * w)
                alto = int(bbox.height * h)
                
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w, x + ancho), min(h, y + alto)
                cara_roi = imagen[y1:y2, x1:x2]
                
                if cara_roi.size > 0:
                    encoding = self.extraer_caracteristicas(cara_roi)
                    nombre, confianza = self.reconocer_cara(encoding)
                    
                    print(f"  Cara {i}: {nombre} - Confianza: {confianza*100:.1f}%")
                    
                    color = (0, 255, 0) if nombre != "Desconocido" else (0, 0, 255)
                    cv2.rectangle(imagen, (x, y), (x + ancho, y + alto), color, 3)
                    
                    etiqueta = f"{nombre} ({confianza*100:.0f}%)"
                    tam_texto = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    
                    cv2.rectangle(imagen, (x, y - 40), (x + tam_texto[0] + 10, y), color, -1)
                    cv2.putText(imagen, etiqueta, (x + 5, y - 12),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow('Resultado del Análisis', imagen)
        print("\nPresiona cualquier tecla para cerrar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Función principal"""
    print("\n" + "="*50)
    print("  SISTEMA DE RECONOCIMIENTO FACIAL")
    print("  Powered by MediaPipe")
    print("="*50)
    
    # Crear sistema
    sistema = ReconocimientoFacial()
    
    # Cargar caras conocidas
    sistema.cargar_caras_conocidas("known_faces")
    
    if not sistema.caras_conocidas:
        print("\n⚠ ATENCIÓN: No hay caras conocidas cargadas")
        print("El sistema funcionará pero todas las caras serán 'Desconocido'\n")
        respuesta = input("¿Continuar de todos modos? (s/n): ")
        if respuesta.lower() != 's':
            print("Saliendo...")
            return
    
    # Menú principal
    while True:
        print("\n" + "="*50)
        print("MENÚ PRINCIPAL")
        print("="*50)
        print("1. 📹 Webcam en tiempo real")
        print("2. 🖼️  Analizar una imagen")
        print("3. 🔄 Recargar caras conocidas")
        print("4. ❌ Salir")
        print("="*50)
        
        opcion = input("\nElige una opción (1-4): ").strip()
        
        if opcion == "1":
            sistema.webcam_tiempo_real()
        
        elif opcion == "2":
            ruta = input("\nRuta de la imagen: ").strip()
            # Remover comillas si las hay
            ruta = ruta.replace('"', '').replace("'", '')
            sistema.analizar_imagen(ruta)
        
        elif opcion == "3":
            sistema.cargar_caras_conocidas("known_faces")
        
        elif opcion == "4":
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("\n⚠ Opción inválida. Intenta de nuevo.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Programa interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()