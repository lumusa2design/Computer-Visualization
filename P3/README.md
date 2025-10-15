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


# Práctica 3 de la asignatura Visión por computador.

<details>
<summary><b>📚 Tabla de contenidos</b></summary>


</details>


## Tarea 1:Los ejemplos ilustrativos anteriores permiten saber el número de monedas presentes en la imagen. ¿Cómo saber la cantidad de dinero presente en ella? Sugerimos identificar de forma interactiva (por ejemplo haciendo clic en la imagen) una moneda de un valor determinado en la imagen (por ejemplo de 1€). Tras obtener esa información y las dimensiones en milímetros de las distintas monedas, realiza una propuesta para estimar la cantidad de dinero en la imagen. Muestra la cuenta de monedas y dinero sobre la imagen. No hay restricciones sobre utilizar medidas geométricas o de color.  
## Una vez resuelto el reto con la imagen ideal proporcionada, captura una o varias imágenes con monedas. Aplica el mismo esquema, tras identificar la moneda del valor determinado, calcula el dinero presente en la imagen. ¿Funciona correctamente? ¿Se observan problemas?

## Nota: Para establecer la correspondencia entre píxeles y milímetros, comentar que la moneda de un euro tiene un diámetro de 23.25 mm. la de 50 céntimos de 24.35, la de 20 céntimos de 22.25, etc. 

## Extras: Considerar que la imagen pueda contener objetos que no son monedas y/o haya solape entre las monedas. Demo en vivo. 

Sobre este primer ejercicio hemos planteado el paradigma sugerido en la tarea, aunque originalmente, para la imagen ideal, se calculaba de forma no-dinámica. Probamos con diferentes imagenes para ver el resultado y esto fallaba, al no tener la misma escala.

Nuestra resolución ha sido guardar en un JSON la calibración (esto con el fin de que el programa pueda tener aplicaciones reales y bastaría con mover a otra celda el codigo que contiene la eliminación o encapsularlo en una función).

Ahor vamos a ir desglosando el código poco a poco:

### Parámetros globales y ajustes

Hemos añadido diferentes constantes que controlan las entradas, la depuración y los elementos del *transform* del Hough. Podemos poner por asi decirlo, tres categorías:
 - **HOUGH**: Sensibilidad del Hough y rango de radios relativos al tamaño de la imagen.
 - **DEDUP**: Los diferentes criterios que hemos usado para fusionar las detecciones casi identicas cercanas.
 - **TWOTONE y CIRC_THRESHOLD**: Umbrales para detectar monedas bicolores y usar la circularidad de la moneda como forma de diferenciar.

 ```python
 import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os 
IMG_PATH = "Monedas.jpg"
CALIB_JSON = "calibracion.json"
COIN_REF = "2 euros"
USE_MPL = False       
SHOW_DEBUG = True
HOUGH_DP = 1.2
HOUGH_PARAM1 = 120
HOUGH_PARAM2 = 50    
HOUGH_MINDIST_F = 0.20  
HOUGH_MINR_F_TIGHT = 0.11
HOUGH_MAXR_F_TIGHT = 0.22
HOUGH_MINR_F_LOOSE = 0.05
HOUGH_MAXR_F_LOOSE = 0.30
HOUGH_MINDIST_F_LOOSE = 0.12
DEDUP_DIST_FRAC = 0.6
DEDUP_RAD_FRAC  = 0.25
TWOTONE_B_THRESH = 6.0    
TWOTONE_LAB_THRESH = 10.0 
CIRC_THRESHOLD = 0.93  
 ```
### Clasificación de monedas con sus diametros oficiales

Mapea através de un diccionario el nombre con su respectivo radio en milímetros.

Se usa el orden para decidir el más cercano en tamaño.

```python
COINS_DIAM_MM = [
    ('1 centimo', 16.25),
    ('2 centimos', 18.75),
    ('10 centimos', 19.75),
    ('5 centimos', 21.25),
    ('20 centimos', 22.25),
    ('1 euro', 23.25),
    ('50 centimos', 24.25),
    ('2 euros', 25.75),
]
COINS_RAD_MM = {name: diam / 2.0 for name, diam in COINS_DIAM_MM}
COIN_ORDER = [name for name, _ in COINS_DIAM_MM]
```
### Preprocesado para Hough

Convierte a escala de grises la imagen y aplica un desenfoque mediante la mediana y una ecualización del histograma para realzar los bordes y contraste de la imagen, facilitando la visualización.

```python
def preprocess_image_for_hough(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return gray, cv2.equalizeHist(cv2.medianBlur(gray, 7))
```

### Detección de círculos usando Hough

- Ajusta el *min/max* radio y distanca mínima en función del tamaño de la imagen
- devuelve círculos como `int`: *(x, y, r)* 

```python
def detect_circles(preprocessed, use_tight=True):
    height, width = preprocessed.shape[:2]
    if use_tight:
        min_radius = int(min(height, width) * HOUGH_MINR_F_TIGHT)
        max_radius = int(min(height, width) * HOUGH_MAXR_F_TIGHT)
        min_dist = int(min(height, width) * HOUGH_MINDIST_F)
    else:
        min_radius = int(min(height, width) * HOUGH_MINR_F_LOOSE)
        max_radius = int(min(height, width) * HOUGH_MAXR_F_LOOSE)
        min_dist = int(min(height, width) * HOUGH_MINDIST_F_LOOSE)
    circles = cv2.HoughCircles(preprocessed, cv2.HOUGH_GRADIENT, dp=HOUGH_DP, minDist=min_dist, param1=HOUGH_PARAM1, param2=HOUGH_PARAM2, minRadius=min_radius, maxRadius=max_radius)
    if circles is None:
        return np.empty((0, 3), dtype=int)
    return np.round(circles[0]).astype(int)
```

### Eliminación de círculos duplicados

Recorre todas las detecciones y fusiona las que estén muy cerca y tienen un radio muy similar.

Se queda con la detección de mayor radiodel grupo.

```python
def deduplicate_circles(circles, distance_fraction=DEDUP_DIST_FRAC, radius_fraction=DEDUP_RAD_FRAC):
    if circles is None or len(circles) == 0:
        return circles
    items = circles.astype(float).tolist()
    used = [False] * len(items)
    result = []
    for i, (x1, y1, r1) in enumerate(items):
        if used[i]:
            continue
        best = (x1, y1, r1)
        for j, (x2, y2, r2) in enumerate(items):
            if j <= i or used[j]:
                continue
            d = np.hypot(x1 - x2, y1 - y2)
            if d < distance_fraction * min(r1, r2) and abs(r1 - r2) < radius_fraction * min(r1, r2):
                if r2 > best[2]:
                    best = (x2, y2, r2)
                used[j] = True
        used[i] = True
        result.append(best)
    return np.array(np.round(result), dtype=int)
```
### Selección de moneda

Debido a algún tipo de incompatibilidad en nuestro entorno virtual, nos hemos visto obligados a usar dos interfaces de dos librerías diferentes. Una usando la de OpenCV y otra la propia de matplotlib. 

Posteriormente, arreglamos el problema, pero por si sucediera en un futuro a la hora de testearlo, hemos dejado las dos opciones.

#### Opción de OpenCV

Díbuja los índices, espera  que el usuario haga click en la moneda de calibración, marca el círculo más cercano y lo devuelve

```python
def select_circle_with_click_cv(image_bgr, circles):
    prompt = "Haz clic sobre la moneda de referencia"
    if circles.size == 0:
        raise RuntimeError("No hay círculos para seleccionar.")
    display = image_bgr.copy()
    for i, (x, y, r) in enumerate(circles):
        cv2.circle(display, (x, y), r, (0, 255, 0), 2)
        cv2.putText(display, f"#{i}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    banner = display.copy()
    h, w = display.shape[:2]
    pad, bar_h = 12, 40
    cv2.rectangle(banner, (0, 0), (w, bar_h + 2 * pad), (0, 0, 0), -1)
    display = cv2.addWeighted(banner, 0.35, display, 0.65, 0.0)
    cv2.putText(display, prompt, (10, 25 + pad // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    selection = {"pt": None}
    def on_mouse(event, mx, my, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            selection["pt"] = (mx, my)
    cv2.namedWindow("Calibracion", cv2.WINDOW_NORMAL)
    cv2.imshow("Calibracion", display)
    cv2.setMouseCallback("Calibracion", on_mouse)
    while True:
        key = cv2.waitKey(20) & 0xFF
        if selection["pt"] is not None:
            break
        if key == 27:
            cv2.destroyWindow("Calibracion")
            raise KeyboardInterrupt("Calibración cancelada por el usuario.")
    mx, my = selection["pt"]
    centers = circles[:, :2].astype(float)
    idx = int(np.argmin(np.linalg.norm(centers - np.array([[mx, my]]), axis=1)))
    x, y, r = tuple(circles[idx])
    cv2.circle(display, (x, y), r, (255, 0, 0), 3)
    cv2.circle(display, (mx, my), 5, (255, 0, 0), -1)
    cv2.imshow("Calibracion", display)
    cv2.waitKey(600)
    cv2.destroyWindow("Calibracion")
    return x, y, r
```

#### Opción con matplotlib

Muestra la imagen y los círculos y con el input recoge el círculo más cercano.

```python
def select_circle_with_click_mpl(image_bgr, circles):
    prompt = "Haz clic sobre la moneda de referencia"
    if circles.size == 0:
        raise RuntimeError("No hay círculos para seleccionar.")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(image_rgb)
    ax.set_title(prompt)
    ax.axis('off')
    for i, (x, y, r) in enumerate(circles):
        ax.add_patch(plt.Circle((x, y), r, fill=False, linewidth=2))
        ax.text(x, y, f"#{i}", fontsize=10, color='r')
    plt.tight_layout()
    pts = plt.ginput(1, timeout=0)
    plt.close(fig)
    if not pts:
        raise KeyboardInterrupt("Calibración cancelada por el usuario.")
    mx, my = pts[0]
    centers = circles[:, :2].astype(float)
    return tuple(circles[int(np.argmin(np.linalg.norm(centers - np.array([[mx, my]]), axis=1)))])
```

### Calculo y guardado de escala
El usuario la moneda que se usará como referencia
Calcula la relación y escala de *px/pmm* y se guarda en `calibracion.json` junto con el tamaño de la imagen.
```python
def calibrate_pixel_scale(image_bgr, reference_coin_name, calib_json=CALIB_JSON, use_mpl=USE_MPL):
    if reference_coin_name not in COINS_RAD_MM:
        raise ValueError(f"Moneda '{reference_coin_name}' no reconocida. Opciones: {list(COINS_RAD_MM.keys())}")
    _, preprocessed = preprocess_image_for_hough(image_bgr)
    circles = deduplicate_circles(detect_circles(preprocessed, use_tight=False))
    if circles.size == 0:
        raise RuntimeError("No se detectaron círculos en calibración. Ajusta parámetros de Hough.")
    picker = select_circle_with_click_mpl if use_mpl else select_circle_with_click_cv
    r_px = float(picker(image_bgr, circles)[2])
    r_mm = COINS_RAD_MM[reference_coin_name]
    scale_px_per_mm = r_px / r_mm
    h, w = image_bgr.shape[:2]
    payload = {"image_size": [h, w], "coin_name": reference_coin_name, "scale_px_per_mm": scale_px_per_mm}
    Path(calib_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return scale_px_per_mm
```
### Carga calibración
Dado que enfocamos este ssitema a un posible caso de uso real, cargamos del jso que servirá de referencia para la calibración.

```python
def load_calibration(calib_json, expected_size=None):
    path = Path(calib_json)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if expected_size and tuple(data.get("image_size", [])) != tuple(expected_size):
        return None
    return data.get("scale_px_per_mm", None)
```
Además sevalida que los tamaños coinciden.

### Rasgos de color y brillo

Usa una máscara anti brillos usando V alto y S bajo

Compara anillos internos con externos para determinar monedas de por ejemplo 1 y 2 euros.

```python
def extract_coin_color_features_lab(image_bgr, circle, inner_frac=0.55, ring_in=0.65, ring_out=0.95, trim=0.15, glare_V=0.88, glare_S=0.22):
    x, y, r = map(int, circle)
    h, w = image_bgr.shape[:2]
    r = max(5, min(r, min(x, w - 1 - x, y, h - 1 - y)))
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    roi = image_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yy, xx = np.ogrid[:lab.shape[0], :lab.shape[1]]
    cx, cy = x - x0, y - y0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask_inner = dist <= (inner_frac * r)
    mask_ring = (dist >= (ring_in * r)) & (dist <= (ring_out * r))
    V = hsv[:, :, 2].astype(np.float32) / 255.0
    S = hsv[:, :, 1].astype(np.float32) / 255.0
    glare_mask = ~((V >= glare_V) & (S <= glare_S))
    def robust_stats(mask):
        mask = mask & glare_mask
        if not np.any(mask):
            return (np.nan, np.nan, np.nan)
        sel = lab[mask].astype(np.float32)
        def trimmed_median(col):
            if col.size == 0:
                return np.nan
            lo = np.quantile(col, trim)
            hi = np.quantile(col, 1.0 - trim)
            col = col[(col >= lo) & (col <= hi)]
            return float(np.median(col)) if col.size > 0 else np.nan
        L = trimmed_median(sel[:, 0])
        a = trimmed_median(sel[:, 1])
        b = trimmed_median(sel[:, 2])
        return (L, a, b)
    Lin, ain, bin_ = robust_stats(mask_inner)
    Lrg, arg, brg = robust_stats(mask_ring)
    delta_b = abs(brg - bin_)
    delta_a = abs(arg - ain)
    delta_lab = float(np.sqrt((Lrg - Lin) ** 2 + (arg - ain) ** 2 + (brg - bin_) ** 2))
    signed_db = float(brg - bin_)
    return {"inner": (Lin, ain, bin_), "ring": (Lrg, arg, brg), "delta_b": delta_b, "delta_a": delta_a, "delta_lab": delta_lab, "signed_db": signed_db}
```

### Medida de cercanía a circulo (circularidad)

Comprueba la cercanía de la figura captada como un círculo a un círculo real. Esto es porque daba errores muchas veces de círculos que no existían.

Si el valor es muy cercano  1 es que es muy circular.

```python
def compute_coin_circularity(image_gray, circle, canny1=80, canny2=160):
    x, y, r = map(int, circle)
    h, w = image_gray.shape[:2]
    x0, x1 = max(0, x - r - 3), min(w, x + r + 4)
    y0, y1 = max(0, y - r - 3), min(h, y + r + 4)
    roi = image_gray[y0:y1, x0:x1]
    if roi.size == 0:
        return np.nan
    edges = cv2.Canny(roi, canny1, canny2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.nan
    cx, cy = x - x0, y - y0
    best_circularity, best_area = np.nan, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        per = cv2.arcLength(cnt, True)
        if per <= 0:
            continue
        circularity = 4 * np.pi * area / (per * per)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        mx, my = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        if np.hypot(mx - cx, my - cy) < 0.25 * r and area > best_area:
            best_area = area
            best_circularity = circularity
    return float(best_circularity)
```

### Umbral adaptativo de dos tonos

Hace al clasificado menos sensible al conjunto de la iluminación percibida en cada foto.

```python
def compute_adaptive_twotone_threshold(image_bgr, circles, default_b=TWOTONE_B_THRESH):
    values = []
    for circle in circles:
        feats = extract_coin_color_features_lab(image_bgr, circle)
        if feats is None or np.isnan(feats["delta_b"]):
            continue
        values.append(feats["delta_b"])
    if len(values) < 3:
        return default_b
    v = np.array(values, dtype=np.float32)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return default_b
    v = np.clip(v, 0, np.percentile(v, 99.5))
    vmax = v.max() + 1e-6
    v8 = (v / vmax * 255).astype(np.uint8).reshape(-1, 1)
    thr8, _ = cv2.threshold(v8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(thr8 / 255.0 * vmax)
    return max(thr, default_b * 0.7)
```
### El valor más cercano

Devuelve el índice más cercano al valor obtenido
```python
def nearest_index(value, array):
    array = np.asarray(array, dtype=float)
    return int(np.argmin(np.abs(array - value)))
```
### Clasificación de la moneda

Gracias a toos los valores obtenidos en el resto de funciones se puede clasificar la moneda gracias a los parámetros. Atendiendo al siguiente orden.

<ol>
    <li>Calcula la moneda gracias al escalado con un escalado</li>
    <li>Busca el radio más cercano</li>
    <li>Si es una moneda de 20 centimos o 1 euro usa el color para comprobar si tiene dos tonos y la diferencia</li>
    <li>Devuelve la moneda</li>
</ol>

Hemos tenido que hacer la diferencia de 1 € y 20 céntimos debido a que, el radio es muy cercano y en muchas imágenes daba error.

```python
def classify_coin(image_bgr, image_gray, circle, scale_px_per_mm, two_tone_b_thresh=None, two_tone_lab_thresh=TWOTONE_LAB_THRESH, circ_thresh=CIRC_THRESHOLD):
    if two_tone_b_thresh is None:
        two_tone_b_thresh = TWOTONE_B_THRESH
    x, y, r_px = circle
    r_mm = r_px / scale_px_per_mm
    candidate_radii = [COINS_RAD_MM[n] for n in COIN_ORDER]
    idx = nearest_index(r_mm, candidate_radii)
    nearest_name = COIN_ORDER[idx]
    diffs = np.abs(np.array(candidate_radii) - r_mm)
    second = np.argsort(diffs)[1] if len(diffs) > 1 else idx
    second_name = COIN_ORDER[second]
    if {nearest_name, second_name} == {"20 centimos", "1 euro"}:
        color = extract_coin_color_features_lab(image_bgr, circle)
        circularity = compute_coin_circularity(image_gray, circle)
        two_tone = False
        if color is not None and np.isfinite(color["delta_b"]):
            two_tone = (color["delta_b"] > two_tone_b_thresh) or (color["delta_lab"] > two_tone_lab_thresh)
            if two_tone and np.isfinite(color["signed_db"]) and color["signed_db"] < 0:
                two_tone = (color["delta_lab"] > (two_tone_lab_thresh + 2.0))
        if two_tone:
            name = "1 euro"
        else:
            if not np.isnan(circularity) and circularity < circ_thresh:
                name = "20 centimos"
            else:
                name = nearest_name
    else:
        name = nearest_name
    return name, r_mm
```

### Dibujo del contorno y las etiquetas
Esta función dibuja el contorno de las monedas y muestra el valor de ellas encima.

```python

def draw_coin_labels(image_bgr, labels):
    out = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    for x, y, r, name, _ in labels:
        cv2.circle(out, (x, y), r, (0, 255, 0), 3)
        cv2.putText(out, name, (x - 40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    return out
```

### Pipeline principal

Flujo del programa para que funcione, en un futuro se podría separar la eliminaciñon del json
```python
if __name__ == "__main__":
    image = cv2.imread(IMG_PATH)
    h, w = image.shape[:2]
    scale = load_calibration(CALIB_JSON, expected_size=(h, w))
    if scale is None:
        scale = calibrate_pixel_scale(image, COIN_REF, calib_json=CALIB_JSON, use_mpl=USE_MPL)
    image_gray, preprocessed = preprocess_image_for_hough(image)
    circles = deduplicate_circles(detect_circles(preprocessed))
    two_tone_b_adapt = compute_adaptive_twotone_threshold(image, circles)
    count_by_type = {n: 0 for n, _, _ in COINS_DIAM_MM}
    labels = []
    for x, y, rpx in circles:
        name, r_mm = classify_coin(image, image_gray, (x, y, rpx), scale, two_tone_b_thresh=two_tone_b_adapt)
        count_by_type[name] += 1
        labels.append((x, y, rpx, name, r_mm))
    vis = draw_coin_labels(image, labels)
    plt.figure(figsize=(7, 10))
    plt.axis("off")
    plt.title(f"Monedas detectadas: {len(circles)}  |  escala: {scale:.2f} px/mm")
    plt.imshow(vis)
    plt.show()
    total = 0.0
    total = 0.0
    for k in COIN_ORDER:
        print(f"{k}: {count_by_type[k]}")
        total += count_by_type[k] * COIN_VALUE[k]
    print(f"Total: {total:.2f} euros")
    if Path(CALIB_JSON).exists():
        os.remove(CALIB_JSON)
```

```mermaid
flowchart TD
    A[Leer imagen (IMG_PATH)] --> B{¿Hay calibración válida?}
    B -- Sí --> C[load_calibration()]
    B -- No --> D[detect_circles(use_tight=false)]
    D --> E[Usuario hace clic en moneda de referencia]
    E --> F[calcular escala px/mm y guardar JSON]
    C --> G[preprocess_image_for_hough()]
    F --> G
    G --> H[detect_circles(use_tight=true)]
    H --> I[deduplicate_circles()]
    I --> J[compute_adaptive_twotone_threshold()]
    I --> K[Clasificar cada círculo: tamaño → {1€,20c} color/circularidad]
    K --> L[draw_coin_labels()]
    L --> M[Mostrar imagen + conteo por tipo]

```

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