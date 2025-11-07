<div style="center">

[![Texto en movimiento](https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=25&duration=1500&pause=9000&color=8A36D2&center=true&vCenter=true&width=400&height=50&lines=Visión+por+computador)]()

---
<div style="center">

[![Abrir Notebook](https://img.shields.io/badge/📘%20Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://github.com/lumusa2design/Computer-Visualization/blob/main/prac1/VC_P1.ipynb)

</div>


---

![Python](https://img.shields.io/badge/python-3.10-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-green?logo=opencv)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Graphs-orange?logo=plotly)

</div>


# Práctica 4 de la asignatura Visión por computador.

<details>
<summary><b>📚 Tabla de contenidos</b></summary>


</details>

 ## Instalación de dependencias
 
Vamos a usar las siguientes librerías:
- `os`: Nos permitirá realizar operaciones de lectura y escritura de ficheros.
- `time`: Con esta librería podremos medir el tiempo y comparar la eficiencia de cada algoritmo.
- `glob`: Realizará la busqueda de archivos que coincidan con una regex.
- `cv2`: Para operar imágenes.
- `numpy`: Nos permitirá operar las imagenes como si fuese una matriz númerica.
- `pandas`: Nos permitirá manejar datos en forma de tablas.
- `YOLO`: Nuestro detector (Es el DOOM de los detectores).

- `Tesseract`: Un OCR que, requiere de instalar de forma externa con una app.
- `EasyOCR`: Otro OCR más ligero y simple.  


 ## Elección del modelo a usar y comparación de dos modelos


Para esta tarea usado de base la práctica 4. Primero decimos comparar que sistemas nos funcionaban mejor, e hicimos un test probando EasyOCR y Tesseract.

```py
import pytesseract
if os.path.isfile(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
```
Fuerza a usar TESSERACT con la ruta dada. Esto es porque daba error al cargarlo, como que no estaba en la ruta por defecto. 

```py
try:
    import torch, easyocr
    use_gpu = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
except:
    reader = None
```

Intenta cargar EasyOCR, activa la GPU si CUDA esta disponible. En caso de que esto falle, sigue sin EasyOCR.

```py
def safe_imread(p):
    a = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(a, cv2.IMREAD_COLOR)
```
Es una lectura más robusta de la imagen, evita errore de Unicode.

```py
def clip(v, lo, hi):
    return max(lo, min(hi, v))
```
Evita lecturas fuera de la imagen

```py
def pad_crop(img, xyxy, pad_pct):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_pct), int(bh * pad_pct)
    x1p = clip(x1 - px, 0, w - 1); y1p = clip(y1 - py, 0, h - 1)
    x2p = clip(x2 + px, 0, w - 1); y2p = clip(y2 + py, 0, h - 1)
    return img[y1p:y2p, x1p:x2p]
```
Expande la caja detectada un porcentaje y recorta. Da margen a la matrícula para OCR.

```py
def preprocess(crop):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 7, 35, 35)
    return cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,5)
```

Pasa a escalas de grises, suaviza bordes preservando contornos y binariza la imagen para homogenizar la iluminación.

```py
def best_box(det):
    if det is None or det.boxes is None or len(det.boxes)==0:
        return None
    conf = det.boxes.conf.cpu().numpy()
    return det.boxes.xyxy.cpu().numpy()[int(np.argmax(conf))]
```

Selecciona la caja con mayor confianza para escoger la matrícula.

```py
def list_imgs(f):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp","*.JPG","*.PNG","*.JPEG","*.WEBP")
    r = []
    for e in exts: r += glob.glob(os.path.join(f,e))
    return sorted(r)
```

Lista las imagenes, compatibles para abrir y usar con el OCR.

### Flujo principal

```
model = YOLO(MODEL_PATH)
out_dir = os.path.join(TEST_DIR,"ocr_preview")
os.makedirs(out_dir, exist_ok=True)
imgs = list_imgs(TEST_DIR)
rows = []
```
- Carga el modelo.
- Crea una carpeta de previsualizaciones.
- Lista las imagenes.
- Prepara una lista para acumular resultados.

### Bucle por imagen

```py
for p in imgs:
    img = safe_imread(p)
    nombre = os.path.basename(p)
    texto_tess, texto_easy = "", ""
    tiempo_tess, tiempo_easy = np.nan, np.nan
    err_tess, err_easy = "", ""
```
- Lee la imagen.
- Inicializa las strings.
- Si da error  devuelve `None`

### Detección con YOLO

```py
det = model.predict(
    img, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES, verbose=False
)[0]
bb = best_box(det)
```
- Realiza la detección.
- Extrae la mejor caja.
- Si no hay caja, la marca como `no_plate`. 


```py
crop = pad_crop(img, bb, PADDING_PCT)
proc = preprocess(crop)
```
Corta y preprocesa la imagen. Para ello amplia la región y Binariza y filtra para el OCR.

```py
t0 = time.perf_counter()
cfg = f'--oem 3 --psm 7 -c tessedit_char_whitelist={ALLOWLIST}'
texto_tess = pytesseract.image_to_string(proc, config=cfg)\
               .strip().upper().replace(" ","").replace("\n","")
tiempo_tess = time.perf_counter() - t0
```

- Cronometra el tiempo que tarda con `time.perf_couter()`.
- Con `--oem 3`: Usa el motor más preciso disponible. Este suee ser LSTM (Usa redes neuronales).
- Usando `--psm 7` (es un modo de configuración de la página de sementación) obtenemos una sola línea de texto.
- Mediante la `tessedit_char_whitelist` se establecen los carácteres permitidos.
- Se limpia de espacios y de saltos de línea. Pasa mayúsculas.

En este caso hemos decidido usar el abecedario entero para que sirva para cualquier coche (debido a que, las matrículas americanas si usan las vocales).

```py
if reader is not None:
    t1 = time.perf_counter()
    res = reader.readtext(proc, detail=0, paragraph=True, allowlist=ALLOWLIST)
    texto_easy = "".join(res).strip().upper().replace(" ","")
    tiempo_easy = time.perf_counter() - t1
else:
    err_easy = "easyocr_not_available"
```
- Devuelve solo texto.
- Cronometra.
- `allowlist`: Restringe carácteres 

```py
vis = crop.copy()
cv2.putText(vis, f"T:{texto_tess}", (10,35), ..., (255,0,0),2)
cv2.putText(vis, f"E:{texto_easy}", (10,70), ..., (0,255,0),2)
cv2.imwrite(os.path.join(out_dir, nombre), vis)
```
Dibuja los textos leídos por cada motor sobre el recorte y guarda una imagen de previsualización.

```py
rows.append({...})
print(f"{nombre} | Tesseract:{texto_tess} | EasyOCR:{texto_easy} | tT:{tiempo_tess:.4f}s | tE:{...:.4f}s")
```
Añade la fila donde se guardan los textos, tiempos y errores.

Imprime el resultado.

```py
df = pd.DataFrame(rows, columns=[...])
out_csv = os.path.join(TEST_DIR,"ocr_comparison.csv")
df.to_csv(out_csv, index=False, encoding="utf-8")
print(f"\nCSV guardado en: {out_csv}")
```
Exporta a CSV.


```py
if __name__ == "__main__":
    main()
```

Ejecuta el código.

## Sistema de detección de códigos de matrículas
El sistema implementado dados los resultados será en EasyOCR ya que nos funcionó mejor que Pytesseract. 
El código desarrollado para la práctica será explicado a continuación:

### 1. Inicialización de Variables y Modelos
```py
detector = YOLO("yolo11n.pt")
plate_model = YOLO(r"C:\Users\luisp\Desktop\VC\prac1\P4\runs\detect\plates_s_1280_rect\weights\best.pt")
```
- `detector`: Carga el modelo YOLO (`yolo11n.pt`) para la detección de vehículos.

- `plate_model`: Carga el modelo entrenado para la detección de matrículas de vehículos (`best.pt`).

### 2. Configuración de Parámetros
```py
TARGET_CLASSES = {"car", "motorbike", "bus", "truck"}
TRACKER = "bytetrack.yaml"
DET_CONF = 0.25
```
- `TARGET_CLASSES`: Define las clases de objetos que el sistema buscará (vehículos de diferentes tipos).

- `TRACKER`: Define el tipo de modelo de seguimiento (en este caso, ByteTrack).

- `DET_CONF`: La confianza mínima requerida para aceptar una detección de objeto (en este caso, 0.25).

### 3. Parámetros de matrículas
```py
PLATE_CONF = 0.28
PLATE_IOU = 0.60
PLATE_IMGSZ = 1280
```
- `PLATE_CONF`: Confianza mínima para la detección de matrículas.

- `PLATE_IOU`: Umbral de Intersección sobre Unión (IoU) para aceptar la detección de la matrícula.

- `PLATE_IMGSZ`: Tamaño de la imagen para el modelo de matrícula.

### 4. Filtrado de Matrículas y Vehículos
```py
PLATE_ONLY_BOTTOM_BAND = True
BOTTOM_FRAC = 0.65
EXTRA_BAND_UP = 0.10
```
- `PLATE_ONLY_BOTTOM_BAND`: Si está habilitado, solo se buscarán matrículas en la parte inferior de la imagen.

- `BOTTOM_FRAC` y `EXTRA_BAND_UP`: Definen el área de la imagen en la que buscar matrículas (en la parte inferior).

### 5. Inicialización de OCR y Variables de Texto
```py
ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PLATE_RE = re.compile(r"^[0-9A-Z]{6,8}$")
reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
```
- `ALLOWLIST`: Conjunto de caracteres permitidos en la matrícula (letras y números).

- `PLATE_RE`: Expresión regular que valida las matrículas de 6 a 8 caracteres.

- `reader`: Inicializa el modelo de OCR (Reconocimiento Óptico de Caracteres) usando EasyOCR, para leer el texto de las matrículas.

### 6. Funciones Auxiliares
La mayoría de funciones que explicaremos a continuación fueron usadas en la P4 anterior ya que es muy parecida a la que estamos realizando, ya que esta Tarea es una extensión de la anterior. Así aquí dejamos un enlace a la práctica anterior donde están explicadas más en detalle en la parte final del README: https://github.com/lumusa2design/Computer-Visualization/tree/main/P4

Así que explicaremos más en detalle las que son nuevas y no las reutilizadas.

`minarect_warp(bgr)`

```py
def minarect_warp(bgr):
    if bgr is None or bgr.size == 0: return None
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    e = cv2.Canny(g, 50, 150)
    e = cv2.dilate(e, np.ones((3,3), np.uint8), 1)
    cnts,_ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    cnt = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    (cx, cy),(w, h),ang = rect
    if w < 20 or h < 10: return None
    box = cv2.boxPoints(rect).astype(np.float32)
    s = box.sum(axis=1); d = np.diff(box, axis=1).ravel()
    tl = box[np.argmin(s)]; br = box[np.argmax(s)]
    tr = box[np.argmin(d)]; bl = box[np.argmax(d)]
    dw, dh = int(max(w, h)), int(min(w, h))
    if w < h: dw, dh = dh, dw
    dst = np.array([[0,0],[dw-1,0],[dw-1,dh-1],[0,dh-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.array([tl,tr,br,bl], dtype=np.float32), dst)
    return cv2.warpPerspective(bgr, M, (dw, dh))
```


Esta función detecta un rectángulo mínimo alrededor de un contorno en una imagen y realiza una transformación de perspectiva para obtener una vista "rectificada" del área de interés (usada principalmente para matrículas).

- Recibe una imagen en color `bgr` (formato BGR).

- Convierte la imagen a escala de grises (`cv2.cvtColor`).

- Aplica un filtro Gaussiano para suavizar la imagen (`cv2.GaussianBlur`).

- Detecta los bordes usando el algoritmo Canny (`cv2.Canny`).

- Realiza una dilatación para aumentar el grosor de los bordes detectados.

- Encuentra los contornos en la imagen con cv2.findContours.

- Selecciona el contorno con el área más grande (`max(cnts, key=cv2.contourArea)`), que se supone que es el objeto de interés.

- Usa `cv2.minAreaRect` para obtener un rectángulo mínimo que se ajusta al contorno.

- Calcula las coordenadas del rectángulo, ajustando la orientación si es necesario.

- Luego, calcula las transformaciones de perspectiva (`cv2.getPerspectiveTransform`) para "rectificar" la imagen y hacerla más fácilmente procesable.

- Devuelve la imagen rectificada.

`preproc_variants(bgr)`

```py
def preproc_variants(bgr):
    if bgr is None or bgr.size == 0: return []
    warped = minarect_warp(bgr)
    roi = warped if warped is not None else bgr
    roi = upscale(roi, 900, 1600)
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    out = []
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    g0 = clahe.apply(g)
    _, th0 = cv2.threshold(g0, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    out.append(th0)
    th1 = cv2.adaptiveThreshold(g0,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,5)
    out.append(th1)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, np.ones((5,5), np.uint8))
    sx = cv2.Sobel(bh, cv2.CV_8U, 1,0,ksize=3)
    _, th2 = cv2.threshold(sx, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    out.append(th2)
    _, th3 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th3 = cv2.morphologyEx(th3, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    out.append(th3)
    return out
```
Esta función genera varias variantes de preprocesamiento de la imagen para mejorar la detección de texto (matrículas) utilizando varias técnicas de umbralización y transformaciones morfológicas.

- Recibe una imagen en color `bgr`.

- Llama a `minarect_warp` para obtener una versión rectificada de la imagen (si es posible).

- Escala la imagen a un tamaño adecuado utilizando la función upscale.

- Convierte la imagen a escala de grises (`cv2.cvtColor`).

- Aplica varias técnicas de procesamiento de imágenes:

- __CLAHE__ (Equalización adaptativa del histograma) para mejorar el contraste.

- __Umbralización binaria__ con el método de Otsu.

- __Umbralización adaptativa__ para obtener una imagen binaria.

- __Morphological blackhat__ para resaltar las estructuras oscuras sobre un fondo brillante.

- __Operaciones de Sobel__ para detectar bordes en la imagen.

- __Apertura morfológica__ para eliminar pequeños ruidos en la imagen.

- Devuelve una lista de imágenes preprocesadas con los diferentes métodos aplicados.


`rotate3(img)`
```py
def rotate3(img):
    h, w = img.shape[:2]
    res = []
    for a in (-5, 0, 5):
        M = cv2.getRotationMatrix2D((w/2, h/2), a, 1.0)
        res.append(cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE))
    return res
```
Esta función rota la imagen en tres ángulos diferentes para ayudar a mejorar los resultados de OCR al manejar posibles variaciones de orientación en las matrículas.

- Recibe una imagen `img`.

- Obtiene las dimensiones de la imagen (`h, w`).

- Rota la imagen en tres ángulos: -5, 0 y 5 grados. Esto se hace para mejorar la precisión del OCR en matrículas que puedan estar ligeramente inclinadas.

- Se utiliza `cv2.getRotationMatrix2D` para obtener la matriz de rotación y `cv2.warpAffine` para aplicar la rotación.

- Devuelve una lista de las imágenes rotadas en los tres ángulos.

`score_text(t)`
```py
def score_text(t):
    s = re.sub(r"\s+", "", t.upper()).replace("O","0") if len(t)<=4 else t.upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    if not s: return "", -1.0
    base = 0.0
    if PLATE_RE.match(s): base += 0.6
    base += max(0.0, 1.0 - abs(len(s)-7)*0.2)
    return s, base
```
Esta función calcula una puntuación de la calidad del texto extraído de la imagen, basándose en la longitud del texto y su formato.

- Recibe un texto `t` .

- Elimina los espacios en blanco y convierte las letras a mayúsculas. Si el texto tiene 4 o menos caracteres, reemplaza las "O" por "0" para mejorar la lectura de matrículas.

- Usa una expresión regular `PLATE_RE` para comprobar si el texto coincide con el formato de una matrícula (6-8 caracteres alfanuméricos).

- Calcula una puntuación base de 0.0 a 1.0:

- Si el texto coincide con el formato de matrícula, se le asigna un bono de 0.6.

- Se da una puntuación adicional basada en la longitud del texto, donde las matrículas de 7 caracteres reciben la mejor puntuación.

- Devuelve el texto normalizado y su puntuación de calidad.

`ocr_best(img_bin)`
```py
def ocr_best(img_bin):
    best_txt, best_sc = "", -1.0
    for rot in rotate3(img_bin):
        try:
            z = reader.readtext(rot, detail=0, paragraph=True, allowlist=ALLOWLIST)
            txt = "".join(z).strip().upper().replace(" ","")
        except:
            txt = ""
        norm, sc = score_text(txt)
        if sc > best_sc:
            best_txt, best_sc = norm, sc
    return best_txt
```
Esta función utiliza OCR para extraer el texto de la imagen binaria (en blanco y negro). Intenta mejorar los resultados al rotar la imagen en varios ángulos.

-  Recibe una imagen binaria `img_bin` (preprocesada).

- Rota la imagen en tres ángulos (utilizando la función `rotate3`).

- Para cada rotación, usa EasyOCR para intentar extraer el texto.

- Para cada texto extraído, calcula una puntuación de calidad utilizando `score_text`.

- Mantiene el texto con la mejor puntuación.

- Devuelve el texto con la mejor puntuación.

`best_box(det)`
```py
def best_box(det):
    if det is None or det.boxes is None or len(det.boxes) == 0: return None
    confs = det.boxes.conf.cpu().numpy()
    return det.boxes.xyxy.cpu().numpy()[int(np.argmax(confs))]
```
Esta función selecciona la mejor caja de predicción de un objeto detectado a partir de la mayor confianza.

- Recibe un objeto `det`, que contiene las predicciones de la detección (probablemente un resultado de YOLO).

- Comprueba si la detección contiene cajas de objetos válidas.

- Obtiene las puntuaciones de confianza (`conf`) de todas las cajas detectadas.

- Devuelve las coordenadas de la caja con la mayor confianza.

- Devuelve las coordenadas de la mejor caja de detección.

### 7. Detección de matrículas
```py
def detect_plate_roi(frame, rx1,ry1,rx2,ry2, vx1,vy1,vx2,vy2):
    crop = frame[ry1:ry2, rx1:rx2]
    if crop.size == 0: return None
    up = upscale(crop, 900, 1600)
    pr = plate_model.predict(source=up, conf=PLATE_CONF, iou=PLATE_IOU, imgsz=PLATE_IMGSZ, max_det=5, augment=True, agnostic_nms=False, verbose=False)
    sx = crop.shape[1]/up.shape[1]; sy = crop.shape[0]/up.shape[0]
    best, best_sc = None, -1e9
    if pr and len(pr[0].boxes)>0:
        for pb in pr[0].boxes:
            px1,py1,px2,py2 = pb.xyxy[0].tolist()
            pconf = float(pb.conf[0].item())
            px1,py1,px2,py2 = int(px1*sx),int(py1*sy),int(px2*sx),int(py2*sy)
            mx1,my1,mx2,my2 = rx1+px1, ry1+py1, rx1+px2, ry1+py2
            w,h = mx2-mx1, my2-my1
            vw,vh = vx2-vx1, vy2-vy1
            if ANTI_HEADLIGHT:
                pc = frame[my1:my2, mx1:mx2]
                if pc.size and is_headlight_like(pc):
                    continue
            if not plausible_plate_absrel(w,h,vw,vh): 
                continue
            if not position_filters(mx1,my1,mx2,my2, vx1,vy1,vx2,vy2):
                continue
            sc = score_plate_abs(mx1,my1,mx2,my2,pconf, vx1,vy1,vx2,vy2)
            if sc > best_sc:
                best_sc, best = sc, (mx1,my1,mx2,my2,pconf)
    if best is None and USE_CONTOUR_FALLBACK:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, None, iterations=1)
        cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        Hc,Wc = crop.shape[:2]
        cand, area_max = None, 0
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if w<20 or h<10: continue
            if x<2 or y<2 or x+w>Wc-2 or y+h>Hc-2: continue
            area = w*h
            if area>area_max:
                cand, area_max = (x,y,x+w,y+h), area
        if cand:
            px1,py1,px2,py2 = cand
            mx1,my1,mx2,my2 = rx1+px1, ry1+py1, rx1+px2, ry1+py2
            w,h = mx2-mx1, my2-my1
            vw,vh = vx2-vx1, vy2-vy1
            if plausible_plate_absrel(w,h,vw,vh) and position_filters(mx1,my1,mx2,my2, vx1,vy1,vx2,vy2):
                best = (mx1,my1,mx2,my2,0.30)
    return best
```
- `detect_plate_roi()`: Recorta una región de la imagen donde se espera que esté la matrícula, la escala, la procesa y luego usa el modelo de matrículas para detectar posibles matrículas en esa región.

### 8. Procesamiento del Video 
Configuración de Archivos y Video
```py
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise FileNotFoundError(VIDEO_IN)

```
- `cv2.VideoCapture()`: Abre el archivo de video de entrada.

Lectura de Frames y Detección de Objetos
```py
gen = detector.track(source=frame, stream=True, persist=True, tracker=TRACKER, conf=DET_CONF, verbose=False)
```
- `detector.track()`: Utiliza el modelo YOLO para detectar objetos en el fotograma y hacer seguimiento con ByteTrack.

Filtrado y Análisis de Objetos Detección
```py
for b in boxes:
    if b.cls is None or b.conf is None or b.xyxy is None:
        continue
    ...

```
- Para cada objeto detectado en el fotograma, se verifica su clase y confianza. Si la clase es válida y supera el umbral de confianza, se procesa para comprobar si es un vehículo.

```py
if name in {"car", "motorbike", "bus", "truck"}:
    ...
```
- Si el objeto detectado es un vehículo, se recorta la región correspondiente y se pasa al modelo de matrículas para detectar posibles matrículas dentro de esa región.

### 9. Almacenamiento de Resultados
```py
cw.writerow([frame_idx, cname, f"{conf:.3f}", tid, x1, y1, x2, y2, plate_flag, f"{plate_conf:.3f}", mx1, my1, mx2, my2, plate_txt])
```
- Escribe los resultados de cada detección en el archivo CSV: el fotograma, la clase del objeto, la confianza, las coordenadas de la caja delimitadora, la matrícula detectada, etc.

### 10. Análisis de Flujo de Objetos
```py
 if FLOW_ANALYSIS:
            ids_to_check = set(list(last_centroid.keys()) + list(inactive_counter.keys()))
            for gid in list(ids_to_check):
                if gid in active_ids:
                    inactive_counter[gid] = 0
                else:
                    inactive_counter[gid] = inactive_counter.get(gid, 0) + 1
                    if inactive_counter[gid] == MISSING_TOLERANCE and gid not in already_counted:
                        cx, cy = last_centroid.get(gid, (None, None))
                        if cx is not None:
                            d = {"left": cx, "right": W-cx, "top": cy, "bottom": H-cy}
                            side = min(d, key=d.get)
                            exit_side_count[side] += 1
                            already_counted.add(gid)
                        last_centroid.pop(gid, None)

        cv2.putText(frame, f"Vehiculos detectados: {len({i for i in unique_vehicle_ids if i!=-1})}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,220,50), 2)
        if collected_plates:
            tail = ",".join(collected_plates[-3:])
            cv2.putText(frame, f"Ultimas placas: {tail}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50,180,255), 2)

        writer.write(frame)
        frame_idx += 1
```
- Realiza un análisis del flujo de objetos, contando cuántos objetos salen por cada borde de la imagen.

### 11. Escribir el Video de Salida
```py
writer.write(frame)
```
- Guarda el fotograma procesado con las anotaciones en el video de salida.
### 12. Resumen Final
```py
with open(PLATES_SUMMARY_TXT, "w", encoding="utf-8") as f:
    f.write(f"Vehiculos unicos: {unique_ids}\n")
    f.write(f"Placas leidas ({len(unique_plates)}):\n")
    for p in unique_plates:
        f.write(p + "\n")
```
- Al finalizar, guarda un resumen en un archivo de texto con la cantidad de vehículos únicos detectados y las matrículas leídas.

### Resumen breve para terminar

El código realizado anteriormente hace lo siguiente:

- Detección de vehículos en un video usando YOLO.

- Detección de matrículas usando un modelo específico para matrículas.

- OCR para leer el texto de las matrículas detectadas.

- Seguimiento de objetos a lo largo del video.

- Almacenamiento de resultados en un archivo CSV y video anotado.

- Análisis de flujo de objetos, identificando cuándo y dónde los vehículos salen del campo de visión.

- Generación de resumen con las matrículas detectadas y los vehículos únicos.


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

- Guía proporcionada por los docentes de la asignatura.
- Chatgpt para corrección de código, guía de instalación y mejora de funciones auxiliares.
- [Reconocimiento de Placas Vehiculares con Python: YOLO, OPENCV y PADDLEOCR.](https://www.youtube.com/watch?v=Ftfwm-0L-c0&t=783s)
- [Ultralytics](https://www.ultralytics.com/es/blog/using-ultralytics-yolo11-for-automatic-number-plate-recognition)
- [Pexels](https://www.pexels.com/es-es/buscar/videos/tr%C3%A1fico/)
