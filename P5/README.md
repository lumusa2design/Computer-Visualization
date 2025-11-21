<div style="center">

[![Texto en movimiento](https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=25&duration=1500&pause=9000&color=8A36D2&center=true&vCenter=true&width=400&height=50&lines=Visión+por+computador)]()

---
<div style="center">

[![Abrir Notebook](https://img.shields.io/badge/📘%20Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://github.com/lumusa2design/Computer-Visualization/blob/main/prac1/VC_P1.ipynb)

</div>


# Práctica 5 de la asignatura Visión por computador.
![Python](https://img.shields.io/badge/python-3.10-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-green?logo=opencv)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Graphs-orange?logo=plotly)


---


</div>

<details>
<summary><b>📚 Tabla de contenidos</b></summary>


</details>

 ## Instalación de dependencias
 
A parte de las pedidas en la práctica instalamos pygames para usar el audiomixer. 

```pip
pip install pygames
```
## Tarea 1

En esta primera tarea hemos realizado un detector de emociones, usando deepface y sus datos biométricos para detectar y diferenciar emociones.

```py
import cv2
import numpy as np
from deepface import DeepFace

def apply_emotion_filter(frame, emotion):
    """Aplica un filtro sencillo según la emoción detectada."""
    emotion = emotion.lower()

    if emotion == "happy":
        # Aumentar saturación (más color)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * 1.5, 0, 255).astype(np.uint8)
        filtered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif emotion == "sad":
        # Escala de grises con tono azulado
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        blue_overlay = np.full_like(gray, (255, 0, 0))  # BGR: azul
        filtered = cv2.addWeighted(gray, 0.7, blue_overlay, 0.3, 0)

    elif emotion == "angry":
        # Tinte rojizo
        red_overlay = np.full_like(frame, (0, 0, 255))  # BGR: rojo
        filtered = cv2.addWeighted(frame, 0.6, red_overlay, 0.4, 0)

    elif emotion == "surprise":
        # Aumentar brillo y contraste
        filtered = cv2.convertScaleAbs(frame, alpha=1.3, beta=25)

    else:
        # Emociones neutras / desconocidas: filtro suave sin cambios fuertes
        filtered = cv2.GaussianBlur(frame, (7, 7), 0)

    return filtered

def run_emotion_filter_demo():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la webcam.")

    print("[INFO] Pulsar ESC para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # DeepFace devuelve una lista de dicts; usamos la primera cara
            obj = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=True
            )
            # En versiones nuevas devuelve lista; en otras, dict
            if isinstance(obj, list):
                dominant_emotion = obj[0]['dominant_emotion']
            else:
                dominant_emotion = obj['dominant_emotion']

            filtered = apply_emotion_filter(frame, dominant_emotion)
            txt = f"Emocion: {dominant_emotion}"
            cv2.putText(filtered, txt, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        except Exception:
            # Si no detecta cara, mostramos el frame original con aviso
            filtered = frame.copy()
            cv2.putText(filtered, "Sin cara / no detectada", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Prototipo 2 - Filtros por emocion", filtered)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


# EJEMPLO DE USO:
run_emotion_filter_demo()

```
Vamos a ir desglosar el código:
```py
import cv2
import numpy as np
from deepface import DeepFace
```

importamos las librerías necesarias:
- `cv2` nos sirve para capturar la webcam y manipular los frames del video.
- `numpy`: para operaciones numéricas y manipular los arrays (recordemos que una imagen en si es un array)
- `Deepface`: sirve para detectar la emoción de la cara de la imagen.

```py
def apply_emotion_filter(frame, emotion):
    """Aplica un filtro sencillo según la emoción detectada."""
    emotion = emotion.lower()
    if emotion == "happy":
        # Aumentar saturación (más color)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * 1.5, 0, 255).astype(np.uint8)
        filtered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
...
```
- `frame`: Imagen que captura la cámara.
- `emotion`: emoción detectada.

Se pasa a formato HSV el color, y cambia el color según la emoción detectada. 

Hemos clasificado las emociones en:
- Feliz
- Triste
- Sorpresa
- Enfadada
- Neutral

```py
def run_emotion_filter_demo():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la webcam.")

    print("[INFO] Pulsar ESC para salir.")
```
Abre la cámara y gestioan posibles errores.

```py
    while True:
        ret, frame = cap.read()
        if not ret:
            break
```
Abre un bucle infinito que lee la cámara.

```py
        try:
            # DeepFace devuelve una lista de dicts; usamos la primera cara
            obj = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=True
            )
            # En versiones nuevas devuelve lista; en otras, dict
            if isinstance(obj, list):
                dominant_emotion = obj[0]['dominant_emotion']
            else:
                dominant_emotion = obj['dominant_emotion']
                            filtered = apply_emotion_filter(frame, dominant_emotion)
            txt = f"Emocion: {dominant_emotion}"
            cv2.putText(filtered, txt, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        except Exception:
            # Si no detecta cara, mostramos el frame original con aviso
            filtered = frame.copy()
            cv2.putText(filtered, "Sin cara / no detectada", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Prototipo 2 - Filtros por emocion", filtered)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    cap.release()
    cv2.destroyAllWindows()
  
# EJEMPLO DE USO:
run_emotion_filter_demo()
```
- Se analiza el frame.
- Se analiza de forma exclusiva la emoción
- Se extrae la emoción dominante
- Aplica el filtro
- Gestiona casos de error
- Muestra la imagen

## Tarea 2:

Para esto hemos hecho un filtro que te pone una máscara de Chayanne.

```py
import cv2
import numpy as np
from deepface import DeepFace 
import pygame

def overlay_on_face(frame, region, overlay, scale=1.4):
    x = region["x"]
    y = region["y"]
    w = int(region["w"] * scale)
    h = int(region["h"] * scale)
    x = x - (w - region["w"]) // 2
    y = y - (h - region["h"]) // 2
    x = max(0, x)
    y = max(0, y)
    h_frame, w_frame = frame.shape[:2]
    w = min(w, w_frame - x)
    h = min(h, h_frame - y)
    if w <= 0 or h <= 0:
        return frame

    overlay_resized = cv2.resize(overlay, (w, h))
    if overlay_resized.shape[2] == 4:
        b, g, r, a = cv2.split(overlay_resized)
        overlay_bgr = cv2.merge((b, g, r))
        alpha = a.astype(float) / 255.0
        alpha = np.stack([alpha, alpha, alpha], axis=-1)
        roi = frame[y:y + h, x:x + w].astype(float)
        blended = alpha * overlay_bgr.astype(float) + (1 - alpha) * roi
        frame[y:y + h, x:x + w] = blended.astype(np.uint8)
    else:
        frame[y:y + h, x:x + w] = overlay_resized

    return frame

def run_face_filter_demo():
    pygame.mixer.init()
    pygame.mixer.music.load("./audio/torero.mp3")

    overlay = cv2.imread("./images/chayanne.webp", cv2.IMREAD_UNCHANGED)
    if overlay is None:
        raise RuntimeError("No se pudo cargar ./images/chayanne.webp")

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la webcam.")

    sonido_activo = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cara_detectada = False

        try:
            obj = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                enforce_detection=True
            )

            if isinstance(obj, list):
                regions = [item["region"] for item in obj]
            else:
                regions = [obj["region"]]

            cara_detectada = True

            filtered = frame.copy()
            for region in regions:
                filtered = overlay_on_face(filtered, region, overlay, scale=1.4)

        except Exception:
            filtered = frame.copy()

        if cara_detectada and not sonido_activo:
            pygame.mixer.music.play()
            sonido_activo = True
        elif not cara_detectada and sonido_activo:
            pygame.mixer.music.stop()
            sonido_activo = False

        cv2.imshow("Filtro Chayanne", filtered)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

run_face_filter_demo()
```

Si vamos desgranando:
```py
import cv2
import numpy as np
from deepface import DeepFace 
import pygame
```
aquí solo usamos una librería extra respecto a la tarea anterior, pygame, que es para reproducir un sonido.

```py
def overlay_on_face(frame, region, overlay, scale=1.4):
    x = region["x"]
    y = region["y"]
    w = int(region["w"] * scale)
    h = int(region["h"] * scale)

        x = x - (w - region["w"]) // 2
    y = y - (h - region["h"]) // 2
    x = max(0, x)
    y = max(0, y)
    h_frame, w_frame = frame.shape[:2]
    w = min(w, w_frame - x)
    h = min(h, h_frame - y)
    if w <= 0 or h <= 0:
        return frame
    overlay_resized = cv2.resize(overlay, (w, h))
    if overlay_resized.shape[2] == 4:
        b, g, r, a = cv2.split(overlay_resized)
        overlay_bgr = cv2.merge((b, g, r))
        alpha = a.astype(float) / 255.0
        alpha = np.stack([alpha, alpha, alpha], axis=-1)
        roi = frame[y:y + h, x:x + w].astype(float)
        blended = alpha * overlay_bgr.astype(float) + (1 - alpha) * roi
        frame[y:y + h, x:x + w] = blended.astype(np.uint8)
    else:
        frame[y:y + h, x:x + w] = overlay_resized

    return frame
```
Los parámetros de entrada:
- `frame`: Imagen de la cámara.
- `region`: Diccionario con la posción de la cara. Propio de deepface.
- `overlay`: La mascara de Chayanne
- `scale`: hace que el overlay sea mas grande.

Después se centra la máscara y se ajusta a la cara ya la perspectiva de la misma. Se redimensiona. En este caso la máscara tiene un canal alpha y por tanto hay que analizar ambos canales.

```py
def run_face_filter_demo():
    pygame.mixer.init()
    pygame.mixer.music.load("./audio/torero.mp3")

    overlay = cv2.imread("./images/chayanne.webp", cv2.IMREAD_UNCHANGED)
    if overlay is None:
        raise RuntimeError("No se pudo cargar ./images/chayanne.webp")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la webcam.")

    sonido_activo = False
```

captura la camara y gestiona el sonido para cuando se ponga la máscara con un flag.

```py
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cara_detectada = False
        try:
            obj = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                enforce_detection=True
            )

            if isinstance(obj, list):
                regions = [item["region"] for item in obj]
            else:
                regions = [obj["region"]]

            cara_detectada = True

            filtered = frame.copy()
            for region in regions:
                filtered = overlay_on_face(filtered, region, overlay, scale=1.4)

        except Exception:
            filtered = frame.copy()
        if cara_detectada and not sonido_activo:
            pygame.mixer.music.play()
            sonido_activo = True
        elif not cara_detectada and sonido_activo:
            pygame.mixer.music.stop()
            sonido_activo = False
```

es el bucle principal que enciende la cámara, gestiona la detección de caras con deepface, la analiza y si detecta una cara y gestiona el sonido según la detecte o no.
 <div align="center">

[![Autor: lumusa2design](https://img.shields.io/badge/Autor-lumusa2design-8A36D2?style=for-the-badge&logo=github&logoColor=white)](https://github.com/lumusa2design)

[![Autor: Nombre2](https://img.shields.io/badge/Autor-guillecab7-6A5ACD?style=for-the-badge&logo=github&logoColor=white)](https://github.com/guillecab7)

[![Docente: Profe](https://img.shields.io/badge/Docente-OTSEDOM-0E7AFE?style=for-the-badge&logo=googlescholar&logoColor=white)](https://github.com/otsedom)

[![Centro: EII](https://img.shields.io/badge/Centro-Escuela%20de%20Ingenier%C3%ADa%20Inform%C3%A1tica-00A86B?style=for-the-badge)](https://www.eii.ulpgc.es/es)

</div>

--- 
## Trabajo realizado

- **Tarea**: `Guillermo y Luis` 
- **README**: `Guillermo y Luis`  

--- 

## Recursos usados

