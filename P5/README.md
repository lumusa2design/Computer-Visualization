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

En esta primera tarea hemos realizado un detector de emociones, entrenando un dataset con SVM extrayendo sus datos biométricos para detectar y diferenciar emociones. Vamos a explicar como fue el entrenamiento.


```py
import os
import pickle
import numpy as np
from deepface import DeepFace
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, classification_report, confusion_matrix


def get_image_paths_by_class(dataset_folder):
    X_paths = []
    y_labels = []
    for class_name in sorted(os.listdir(dataset_folder)):
        class_dir = os.path.join(dataset_folder, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            X_paths.append(os.path.join(class_dir, fname))
            y_labels.append(class_name)
    return X_paths, y_labels


def compute_embeddings(image_paths, model_name="Facenet"):
    embeddings = []
    valid_labels_idx = []  # índices de las imágenes que sí se han podido procesar
    for idx, path in enumerate(image_paths):
        try:
            emb_obj = DeepFace.represent(
                img_path=path,
                model_name=model_name,
                enforce_detection=False  # asumimos que hay cara, pero no queremos que casque
            )
            embeddings.append(emb_obj[0]["embedding"])
            valid_labels_idx.append(idx)
        except Exception as e:
            print(f"[AVISO] No se pudo procesar {path}: {e}")
    if len(embeddings) == 0:
        raise RuntimeError("No se pudo obtener ningún embedding. Revisa el dataset.")
    return np.asarray(embeddings, dtype="float32"), valid_labels_idx


def train_emotion_svm(train_folder, model_output_path, n_splits=5, model_name="Facenet"):
    # 1) Cargar rutas e etiquetas
    X_paths, y_labels = get_image_paths_by_class(train_folder)
    if len(X_paths) == 0:
        raise RuntimeError("El dataset de entrenamiento está vacío o las rutas son incorrectas.")

    # 2) Embeddings
    print("[INFO] Calculando embeddings de entrenamiento con DeepFace...")
    X, valid_idx = compute_embeddings(X_paths, model_name=model_name)
    y_labels = [y_labels[i] for i in valid_idx]

    # 3) Codificar etiquetas
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    # 4) Escalado + SVM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = SVC(kernel="rbf", probability=True, class_weight="balanced")

    # 5) Validación cruzada
    print("[INFO] Validación cruzada (StratifiedKFold)...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    precs = []
    recs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), start=1):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_val, y_pred, average="weighted", zero_division=0)
        precs.append(prec)
        recs.append(rec)

        print(f"\n=== Fold {fold} ===")
        print(f"Precisión (weighted): {prec:.3f}")
        print(f"Recall    (weighted): {rec:.3f}")
        print("\nClassification report:")
        print(classification_report(y_val, y_pred, target_names=le.classes_, zero_division=0))
        print("Matriz de confusión:")
        print(confusion_matrix(y_val, y_pred, labels=range(len(le.classes_))))

    print("\n=== Medias sobre los folds (entrenamiento) ===")
    print(f"Precisión media: {np.mean(precs):.3f}")
    print(f"Recall medio:    {np.mean(recs):.3f}")

    # 6) Entrenar modelo final con todos los datos de entrenamiento
    print("\n[INFO] Entrenando modelo final con todo el conjunto de entrenamiento...")
    clf.fit(X_scaled, y)

    # 7) Guardar a disco
    model = {
        "scaler": scaler,
        "label_encoder": le,
        "classifier": clf,
        "deepface_model_name": model_name
    }
    with open(model_output_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[INFO] Modelo guardado en: {model_output_path}")


def evaluate_on_folder(test_folder, model_path):
    # 1) Cargar modelo
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    scaler = model["scaler"]
    le = model["label_encoder"]
    clf = model["classifier"]
    model_name = model.get("deepface_model_name", "Facenet")

    # 2) Cargar rutas e etiquetas reales
    X_paths, y_labels = get_image_paths_by_class(test_folder)
    if len(X_paths) == 0:
        raise RuntimeError("El dataset de test está vacío o las rutas son incorrectas.")

    print("[INFO] Calculando embeddings de test con DeepFace...")
    X_test, valid_idx = compute_embeddings(X_paths, model_name=model_name)
    y_labels = [y_labels[i] for i in valid_idx]
    y_true = le.transform(y_labels)

    # 3) Escalar y predecir
    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)

    # 4) Métricas
    print("\n=== Evaluación en el conjunto de test ===")
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))
    print("Matriz de confusión:")
    print(confusion_matrix(y_true, y_pred, labels=range(len(le.classes_))))


if __name__ == "__main__":
    DATA_ROOT = r"C:\Users\luisp\Desktop\VC\emotions"

    TRAIN_DIR = os.path.join(DATA_ROOT, "train")  # aquí entrenas
    TEST_DIR  = os.path.join(DATA_ROOT, "test")        # aquí evalúas

    MODEL_PATH = "modelo_emociones_svm.pkl"

    # Entrenar
    train_emotion_svm(TRAIN_DIR, MODEL_PATH, n_splits=5, model_name="Facenet")

    # Evaluar
    evaluate_on_folder(TEST_DIR, MODEL_PATH)

```
### 1. Función `get_image_paths_by_class`
```py
def get_image_paths_by_class(dataset_folder):
    X_paths = []
    y_labels = []
    for class_name in sorted(os.listdir(dataset_folder)):
        class_dir = os.path.join(dataset_folder, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            X_paths.append(os.path.join(class_dir, fname))
            y_labels.append(class_name)
    return X_paths, y_labels
```
- Recorre las subcarpetas de `dataset_folder` (cada subcarpeta = una clase/emoción).

- Para cada imagen `.jpg/.jpeg/.png`:

    - Guarda la ruta en `X_paths`.

    - Guarda el nombre de la carpeta (la clase) en `y_labels`.

- Datos devueltos:

    - `X_paths`: lista de rutas de imagen.

    - `y_labels`: lista de etiquetas de clase (strings).


### 2. Función `compute_embeddings`
```py
def compute_embeddings(image_paths, model_name="Facenet"):
    embeddings = []
    valid_labels_idx = []
    for idx, path in enumerate(image_paths):
        try:
            emb_obj = DeepFace.represent(
                img_path=path,
                model_name=model_name,
                enforce_detection=False
            )
            embeddings.append(emb_obj[0]["embedding"])
            valid_labels_idx.append(idx)
        except Exception as e:
            print(f"[AVISO] No se pudo procesar {path}: {e}")
    if len(embeddings) == 0:
        raise RuntimeError("No se pudo obtener ningún embedding. Revisa el dataset.")
    return np.asarray(embeddings, dtype="float32"), valid_labels_idx
```
- Para cada imagen en `image_paths` llama a `DeepFace.represen`t con `model_name` (por defecto Facenet).

- `enforce_detection=False`: si no detecta cara, no revienta, pero puede devolver algo “raro”.

- Extrae el vector de características `["embedding"]` y lo añade a `embeddings`.

- Si alguna imagen falla, lo captura en un `try/except` y solo imprime un aviso.

- `embeddings`: array N x D de floats (N = nº de imágenes procesadas, D = dimensión del embedding).

- `valid_labels_idx`: índices de las imágenes que sí se pudieron procesar (para filtrar las etiquetas).

### 3. Función `train_emotion_svm`
```py
def train_emotion_svm(train_folder, model_output_path, n_splits=5, model_name="Facenet"):
    # 1) Cargar rutas e etiquetas
    X_paths, y_labels = get_image_paths_by_class(train_folder)
    if len(X_paths) == 0:
        raise RuntimeError("El dataset de entrenamiento está vacío o las rutas son incorrectas.")

    # 2) Embeddings
    print("[INFO] Calculando embeddings de entrenamiento con DeepFace...")
    X, valid_idx = compute_embeddings(X_paths, model_name=model_name)
    y_labels = [y_labels[i] for i in valid_idx]

    # 3) Codificar etiquetas
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    # 4) Escalado + SVM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = SVC(kernel="rbf", probability=True, class_weight="balanced")

    # 5) Validación cruzada
    print("[INFO] Validación cruzada (StratifiedKFold)...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    precs = []
    recs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), start=1):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_val, y_pred, average="weighted", zero_division=0)
        precs.append(prec)
        recs.append(rec)

        print(f"\n=== Fold {fold} ===")
        print(f"Precisión (weighted): {prec:.3f}")
        print(f"Recall    (weighted): {rec:.3f}")
        print("\nClassification report:")
        print(classification_report(y_val, y_pred, target_names=le.classes_, zero_division=0))
        print("Matriz de confusión:")
        print(confusion_matrix(y_val, y_pred, labels=range(len(le.classes_))))

    print("\n=== Medias sobre los folds (entrenamiento) ===")
    print(f"Precisión media: {np.mean(precs):.3f}")
    print(f"Recall medio:    {np.mean(recs):.3f}")

    # 6) Entrenar modelo final con todos los datos de entrenamiento
    print("\n[INFO] Entrenando modelo final con todo el conjunto de entrenamiento...")
    clf.fit(X_scaled, y)

    # 7) Guardar a disco
    model = {
        "scaler": scaler,
        "label_encoder": le,
        "classifier": clf,
        "deepface_model_name": model_name
    }
    with open(model_output_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[INFO] Modelo guardado en: {model_output_path}")
```
#### 3.1 Cargar rutas y etiquetas

- Obtiene todas las rutas e etiquetas del conjunto de entrenamiento.

- Si no hay imágenes, lanza error.

#### 3.2 Calcular embeddings

- Calcula los embeddings `X`.

- Filtra `y_labels` para quedarse solo con las que tienen embedding válido.

#### 3.3 Codificar etiquetas

- Convierte las etiquetas de texto (p.ej. "happy", "sad") a enteros (0, 1, 2, …).

- Guarda el `LabelEncoder` para usarlo luego en test.

#### 3.4 Escalado + definición del SVM

- StandardScaler: centra y escala los embeddings (media 0, varianza 1) → importante para SVM.

- SVC con:

    - kernel="rbf": kernel gaussiano.

    - probability=True: permite predict_proba.

    - class_weight="balanced": ajusta el peso de cada clase según su frecuencia (para datasets desbalanceados).

#### 3.5 Validación Cruzada

- Usa StratifiedKFold para mantener la proporción de clases en cada fold.

- Divide datos en train/valid por fold.

- Entrena el SVM en X_train, y_train.

- Evalúa en X_val, y_val.

- Calcula precisión y recall ponderados (average="weighted").

- Imprime:

    - precisión y recall por fold.

    - classification_report por clase.

    - matriz de confusión.

#### 3.6 Entrenar modelo final y guardar

- Reentrena el SVM usando todos los datos de entrenamiento (no solo un fold).

- Guarda en un pickle:

    - el escalador,

    - el codificador de etiquetas,

    - el clasificador entrenado,

    - el nombre del modelo de DeepFace utilizado.

### 4. Función evaluate_on_folder
```py
ef evaluate_on_folder(test_folder, model_path):
    # 1) Cargar modelo
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    scaler = model["scaler"]
    le = model["label_encoder"]
    clf = model["classifier"]
    model_name = model.get("deepface_model_name", "Facenet")

    # 2) Cargar rutas e etiquetas reales
    X_paths, y_labels = get_image_paths_by_class(test_folder)
    if len(X_paths) == 0:
        raise RuntimeError("El dataset de test está vacío o las rutas son incorrectas.")

    print("[INFO] Calculando embeddings de test con DeepFace...")
    X_test, valid_idx = compute_embeddings(X_paths, model_name=model_name)
    y_labels = [y_labels[i] for i in valid_idx]
    y_true = le.transform(y_labels)

    # 3) Escalar y predecir
    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)

    # 4) Métricas
    print("\n=== Evaluación en el conjunto de test ===")
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))
    print("Matriz de confusión:")
    print(confusion_matrix(y_true, y_pred, labels=range(len(le.classes_))))
```
#### 4.1 Cargar modelo

- Lee del disco todo lo que guardaste.

- Si no encuentra el nombre del modelo, usa "Facenet" por defecto.

#### 4.2 Cargar y procesar imágenes de test

- Obtiene rutas y etiquetas del conjunto de test.

- Calcula embeddings con el mismo modelo DeepFace que en entrenamiento.

- Filtra etiquetas según valid_idx.

- y_true: etiquetas reales en forma numérica, usando el mismo LabelEncoder.

#### 4.3 Escalar, predecir y evaluar

- Aplica el mismo escalado que en entrenamiento.

- Predice con el SVM.

- Imprime informe de clasificación y matriz de confusión sobre el conjunto de test.


### 5. Bloque principal (`if __name__ == "__main__":`)
```py
if __name__ == "__main__":
    DATA_ROOT = r"C:\Users\luisp\Desktop\VC\emotions"

    TRAIN_DIR = os.path.join(DATA_ROOT, "train")  # aquí entrenas
    TEST_DIR  = os.path.join(DATA_ROOT, "test")        # aquí evalúas

    MODEL_PATH = "modelo_emociones_svm.pkl"

    # Entrenar
    train_emotion_svm(TRAIN_DIR, MODEL_PATH, n_splits=5, model_name="Facenet")

    # Evaluar
    evaluate_on_folder(TEST_DIR, MODEL_PATH)

```
- Define la ruta base del dataset.

- Especifica carpetas de train y test.

- Define dónde guardar el modelo (modelo_emociones_svm.pkl).

- Llama primero a train_emotion_svm, luego a evaluate_on_folder.

En resumen: El script recorre las carpetas de emociones, extrae embeddings faciales con DeepFace (Facenet), los escala, entrena un SVM con validación cruzada estratificada, guarda el modelo (escalador + encoder + SVM) y después carga ese modelo para evaluar automáticamente en un conjunto de test, mostrando informe de clasificación y matriz de confusión.

### Resultados del entrenamiento 
Los resultados del entrenamiento han sido los siguientes:

=== Fold 1 ===

Precisión (weighted): 0.532

Recall    (weighted): 0.524

Classification report:

              precision    recall  f1-score   support

    angry       0.44      0.50      0.47       799

    disgusted       0.56      0.52      0.54        88

    fearful       0.44      0.32      0.37       819

    happy       0.71      0.62      0.66      1443

    neutral       0.52      0.52      0.52       993

        sad       0.40      0.49      0.44       966

    surprised       0.57      0.66      0.61       634
    
Resumen: 

    accuracy                           0.52      5742

    macro avg       0.52      0.52      0.52      5742

    weighted avg       0.53      0.52      0.52      5742

Matriz de confusión:

    [[399  15  61  62  87 140  35]

    [ 21  46   5   5   1  10   0]
    
    [ 113    6   72 1086  185  230   82]

    [ 129    4   66  139  614  198   83]

    [ 180   17  109   98  179  598   66]

    [  42    4   70   43   42   64  566]]



A continuación vemos el código donde probamos el modelo entrenado, hemos de comentar que funciona mucho mejor el modelo de deepface, pero si detecta algunas emociones, auqnue no de manera estable.

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

- Apuntes del profesorado
- [Doc de pygame](https://www.pygame.org/docs/ref/mixer.html)
- [deepface de Pypi](https://pypi.org/project/deepface/)
- ChatGPT como ajustador de parámetros y debuggeador.