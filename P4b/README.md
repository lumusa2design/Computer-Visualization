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


 ## Tarea 4.1:


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

pasa a escalas de grises, suaviza bordes preservando contornos y binariza la imagen para homogenizar la iluminación.

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


---



 <div align="center">

[![Autor: lumusa2design](https://img.shields.io/badge/Autor-lumusa2design-8A36D2?style=for-the-badge&logo=github&logoColor=white)](https://github.com/lumusa2design)

[![Autor: Nombre2](https://img.shields.io/badge/Autor-guillecab7-6A5ACD?style=for-the-badge&logo=github&logoColor=white)](https://github.com/guillecab7)

[![Docente: Profe](https://img.shields.io/badge/Docente-OTSEDOM-0E7AFE?style=for-the-badge&logo=googlescholar&logoColor=white)](https://github.com/otsedom)

[![Centro: EII](https://img.shields.io/badge/Centro-Escuela%20de%20Ingenier%C3%ADa%20Inform%C3%A1tica-00A86B?style=for-the-badge)](https://www.eii.ulpgc.es/es)

</div>

--- 
## Trabajo realizado

- **Tarea 1**: `Guillermo`
- **Tarea 2**: `Guillermo (Sobel) y Luis (Gráfica)` 
- **Tarea 3**: `Luis` 
- **Tarea 4**: `Guillermo y Luis` 
- **README**: `Guillermo y Luis`  

--- 

## Recursos usados

- Guía proporcionada por los docentes de la asignatura.
- Chatgpt para corrección de código, guía de instalación y mejora de funciones auxiliares.
- [Reconocimiento de Placas Vehiculares con Python: YOLO, OPENCV y PADDLEOCR.](https://www.youtube.com/watch?v=Ftfwm-0L-c0&t=783s)
- [Ultralytics](https://www.ultralytics.com/es/blog/using-ultralytics-yolo11-for-automatic-number-plate-recognition)
- [Pexels](https://www.pexels.com/es-es/buscar/videos/tr%C3%A1fico/)
