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

 ## Tarea a realizar
 La tarea consiste en desarrollar un prototipo que procese un vídeo de ejemplo proporcionado y varios vídeos inluídos para pruebas:

Los aspectos que debe ralizar dicho prototipo son:

- Detecte y siga las personas y vehículos presentes

- Detecte las matrículas de los vehículos presentes

- Cuente el total de cada clase

- Vuelque a disco un vídeo que visualice los resultados

- Genere un archivo csv con el resultado de la detección y seguimiento

Para la realización de esta tarea hemos recopilado entre varios miembros de la clase un dataset de matrículas de vehículos. Además tenemos dos scripts para llevar a cabo el entrenamiento, validación y test, y para organizar en carpetas todos los archivos que recopilamos. Estos fueron realizados entre los participantes de dicho recopilatorio y son division_archivos.py y json_a_txt.py.

## Descripción de los archivos implementados
**`json_a_txt.py`**

Este código tiene como objetivo convertir las anotaciones en formato JSON generadas por la herramienta labelme a un formato adecuado para YOLO (un formato comúnmente utilizado en tareas de detección de objetos).
Aquí está el desglose paso a paso de lo que hace el código:

1. **Importación de bibliotecas**
```python
import json
import os
import glob
from PIL import Image  # Necesitas: pip install Pillow
```
- json: Se usa para cargar y manipular archivos JSON.

- os: Proporciona una interfaz para interactuar con el sistema operativo, como gestionar archivos y directorios.

- glob: Se usa para buscar archivos que coincidan con un patrón específico (en este caso, archivos .json).

- PIL (Pillow): Se usa para trabajar con imágenes, especialmente para obtener las dimensiones de las imágenes.

2. **Configuración de rutas y mapeo de clases**
```python
CLASS_MAPPING = {
    "plate": 0
}
SOURCE_JSON_DIR = r"C:\Users\luisp\Desktop\VC\prac1\UC3M-LP\test"
SOURCE_IMAGES_DIR = r"C:\Users\luisp\Desktop\VC\prac1\UC3M-LP\test"
OUTPUT_TXT_DIR = r"C:\Users\luisp\Desktop\VC\prac1\P4\plates\txt"
```
- CLASS_MAPPING: Un diccionario que mapea las etiquetas de las clases a sus respectivos identificadores (ID). En este caso, "plate" se mapea al ID 0.

- SOURCE_JSON_DIR: Ruta donde se encuentran los archivos JSON de las anotaciones.

- SOURCE_IMAGES_DIR: Ruta donde se encuentran las imágenes correspondientes a los archivos JSON.

- OUTPUT_TXT_DIR: Ruta donde se guardarán los archivos de salida con las anotaciones en formato YOLO.

3.  **Función `convert_labelme_json_to_yolo`**
```python
def convert_labelme_json_to_yolo(json_path, image_width, image_height):
```
Esta función convierte un archivo JSON en formato labelme a una lista de líneas en formato YOLO (que contiene la clase e información de la caja delimitadora normalizada).

_Descripción de los pasos dentro de esta función_:

- Lectura del archivo JSON: Se abre el archivo JSON y se carga su contenido. Si ocurre un error, se imprime un mensaje de error.
```python
with open(json_path, 'r') as f:
    data = json.load(f)
```
- Iterar sobre las formas (shapes): En el archivo JSON, cada anotación de objeto está representada por un "shape". Se obtiene el nombre de la clase (label_name) y se verifica si está en el diccionario CLASS_MAPPING. Si no está en el mapeo, se omite la forma.
```python
for shape in data.get("shapes", []):
    label_name = shape.get("label")
    if label_name not in CLASS_MAPPING:
        continue
    class_id = CLASS_MAPPING[label_name]
```
- Obtener las coordenadas: Dependiendo del tipo de forma (rectángulo o polígono), se obtienen las coordenadas de los puntos. Si es un rectángulo, se usan dos puntos (esquinas opuestas), y si es un polígono, se calcula el cuadro delimitador más pequeño que rodea el polígono.
```python
shape_type = shape.get("shape_type")
points = shape.get("points", [])
```
- Cálculo de la caja delimitadora:

Para el tipo rectángulo, se utilizan los dos puntos dados para calcular las esquinas.

Para el tipo polígono, se calcula el mínimo y máximo de las coordenadas x y y de todos los puntos del polígono.
```python
if shape_type == "rectangle":
    # Calcula las coordenadas de un rectángulo
elif shape_type == "polygon":
    # Calcula las coordenadas de un polígono
```
- Conversión a formato YOLO: El formato YOLO requiere que las coordenadas de la caja delimitadora estén normalizadas con respecto al tamaño de la imagen. Esto significa dividir las coordenadas por el ancho y alto de la imagen para obtener valores entre 0 y 1.
```python
box_width = x2 - x1
box_height = y2 - y1
x_center = x1 + (box_width / 2)
y_center = y1 + (box_height / 2)

x_center_norm = x_center / image_width
y_center_norm = y_center / image_height
width_norm = box_width / image_width
height_norm = box_height / image_height
```
- Generación de la línea en formato YOLO: La línea generada tiene el formato `<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>`.
```python
yolo_lines.append(f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}")
```

4. **Función `get_image_dimensions`**
```python
def get_image_dimensions(image_name_base, images_dir):
```
Esta función se encarga de obtener las dimensiones de la imagen (ancho y alto) dada su base de nombre, buscando la imagen en la carpeta de imágenes proporcionada. Soporta formatos .jpg, .jpeg y .png.

```python
img_filename = image_name_base + ext
path = os.path.join(images_dir, img_filename)
```
Utiliza la librería Pillow para abrir la imagen y obtener sus dimensiones.

```python
with Image.open(image_path) as img:
    return img.width, img.height
```
5. **Función `main`**
```python
def main():
```
Esta es la función principal que se ejecuta cuando el script se corre. Aquí se realiza todo el flujo del procesamiento:

Crear la carpeta de salida para los archivos .txt con las anotaciones en formato YOLO.
```python
os.makedirs(OUTPUT_TXT_DIR, exist_ok=True)
```
Buscar todos los archivos JSON en el directorio de entrada.
```python
json_files = glob.glob(os.path.join(SOURCE_JSON_DIR, "*.json"))
```
Leer cada archivo JSON y procesarlo:

- Obtener las dimensiones de la imagen correspondiente.

- Convertir las anotaciones de cada archivo JSON a formato YOLO usando la función convert_labelme_json_to_yolo.

- Guardar las anotaciones convertidas en un archivo .txt con el mismo nombre que la imagen.

```python
for json_path in json_files:
    # Obtener las dimensiones de la imagen
    width, height = get_image_dimensions(img_name_only, SOURCE_IMAGES_DIR)
    # Convertir las anotaciones a formato YOLO
    yolo_data_lines = convert_labelme_json_to_yolo(json_path, width, height)
    # Escribir el archivo .txt
    output_txt_path = os.path.join(OUTPUT_TXT_DIR, img_name_only + ".txt")
```

6. **Resultado**

Finalmente, el script muestra cuántos archivos han sido convertidos exitosamente y almacenados en la carpeta de salida.

```python
print(f"\nConversión completada. Se generaron {converted_count} archivos .txt en '{OUTPUT_TXT_DIR}'")
```


**`division_archivos.py`**

Este script tiene como objetivo organizar un conjunto de imágenes y sus correspondientes archivos de etiquetas en tres conjuntos separados: **Entrenamiento (train)**, **Validación (val)** y **Prueba (test)**. La estructura de carpetas generada sigue el formato requerido para modelos de detección de objetos, como YOLO. Además, se realiza la división aleatoria de los datos en las proporciones especificadas (80% para entrenamiento + validación, 20% para prueba).

#### Estructura de Carpetas

El script organiza los archivos en la siguiente estructura de directorios:


- **train/images/**: Contiene las imágenes del conjunto de entrenamiento.
- **train/labels/**: Contiene los archivos de etiquetas correspondientes a las imágenes de entrenamiento.
- **val/images/**: Contiene las imágenes del conjunto de validación.
- **val/labels/**: Contiene los archivos de etiquetas correspondientes a las imágenes de validación.
- **test/images/**: Contiene las imágenes del conjunto de prueba.
- **test/labels/**: Contiene los archivos de etiquetas correspondientes a las imágenes de prueba.

#### Configuración

Antes de ejecutar el script, debes ajustar las rutas de las carpetas que contienen las imágenes y las etiquetas. En nuestro caso fue la siguiente:

```python
# --- 1. CONFIGURACIÓN ---

# Ruta a la carpeta que contiene TODAS tus imágenes originales
SOURCE_IMAGES_DIR = "C:\\Users\\luisp\\Desktop\\VC\\prac1\\P4\\plates"
SOURCE_LABELS_DIR = "C:\\Users\\luisp\\Desktop\\VC\\prac1\\P4\\plates\\txt"


# Carpeta de salida donde se creará la estructura train/val/test
OUTPUT_DIR = "matriculas" 

# Proporciones de reparto (basado en la práctica)
# 20% para Test
TEST_RATIO = 0.20
# Del 80% restante, 20% para Validación (es decir, 16% del total)
VAL_RATIO = 0.20 

# Semilla para aleatoriedad (para que el reparto sea reproducible)
RANDOM_SEED = 42
# ---------------------------
```

En cuanto a las funciones del Script, tenemos las siguientes:

**1. `crear_estructura_directorios(base_path)`**

Esta función crea la estructura de carpetas necesaria para almacenar las imágenes y las etiquetas en los conjuntos de entrenamiento, validación y prueba.

```python
def crear_estructura_directorios(base_path):
    """
    Crea la estructura de carpetas requerida por YOLO:
    - base_path/
        - train/
            - images/
            - labels/
        - val/
            - images/
            - labels/
        - test/
            - images/
            - labels/
    """
    sets = ["train", "val", "test"]
    sub_dirs = ["images", "labels"]
    for s in sets:
        for sub in sub_dirs:
            # os.path.join crea la ruta de forma segura (ej. "dataset_yolo/train/images")
            path = os.path.join(base_path, s, sub)
            # exist_ok=True evita errores si las carpetas ya existen
            os.makedirs(path, exist_ok=True)
    print(f"Estructura de carpetas creada en '{base_path}'")
```
Se crea un directorio base (matriculas/) con las subcarpetas correspondientes a train, val y test, cada uno con subcarpetas images y labels.


**2. `copiar_archivos(lista_archivos, set_name)`**

Esta función copia los archivos de imagen y sus correspondientes archivos de etiquetas a las carpetas correspondientes dentro de los conjuntos train, val o test. Asegura que las imágenes y sus etiquetas se emparejen correctamente.

```python
def copiar_archivos(lista_archivos, set_name):
    """
    Copia un par de archivos (imagen + label) a su carpeta de destino.
    """
    count = 0
    for img_source_path in lista_archivos:
        try:
            # 1. Obtener el nombre base del archivo (sin extensión)
            file_basename = os.path.basename(img_source_path)
            file_name, _ = os.path.splitext(file_basename)
            
            # 2. Definir la ruta del label (.txt) correspondiente
            label_source_path = os.path.join(SOURCE_LABELS_DIR, file_name + ".txt")
            
            # 3. Definir rutas de destino
            img_dest_path = os.path.join(OUTPUT_DIR, set_name, "images", file_basename)
            label_dest_path = os.path.join(OUTPUT_DIR, set_name, "labels", file_name + ".txt")
            
            # 4. Comprobar que el label existe
            if not os.path.exists(label_source_path):
                print(f"¡Aviso! No se encontró label para {file_basename}. Omitiendo este archivo.")
                continue
                
            # 5. Copiar los archivos
            shutil.copy(img_source_path, img_dest_path)
            shutil.copy(label_source_path, label_dest_path)
            count += 1
        except Exception as e:
            print(f"Error copiando {img_source_path}: {e}")
            
    print(f"Copiados {count} pares de archivos (img+label) a '{set_name}'")
```

**3. `main()`**

La función principal (main) organiza todo el flujo del script. Realiza los siguientes pasos:

Establece una semilla aleatoria para asegurar que el reparto de los archivos sea reproducible.

```python
    # Establecer la semilla para reproducibilidad
    random.seed(RANDOM_SEED)
```

Crea la estructura de directorios de salida.

```python
    # Crear la estructura de carpetas de salida
    crear_estructura_directorios(OUTPUT_DIR)
```


Recopila todas las imágenes en los formatos .jpg, .jpeg y .png desde la carpeta de imágenes de entrada.

```python
# 1. Encontrar todas las imágenes (soporta .jpg, .jpeg, .png)
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    all_image_files = []
    for ext in image_extensions:
        all_image_files.extend(glob.glob(os.path.join(SOURCE_IMAGES_DIR, ext)))
    
    if not all_image_files:
        print(f"Error: No se encontraron imágenes en {SOURCE_IMAGES_DIR}")
        return
```

Mezcla aleatoriamente la lista de imágenes.

```python
 # 2. Mezclar la lista de forma aleatoria
    random.shuffle(all_image_files)
    
    total_files = len(all_image_files)
    print(f"\nTotal de {total_files} imágenes encontradas.")
```


Calcula los índices de división para repartir los archivos en tres conjuntos: train, val y test, según las proporciones definidas.

```python
 # 3. Calcular los índices de división
    # División 80/20 (Train+Val / Test)
    test_split_index = int(total_files * (1 - TEST_RATIO))
    
    test_files = all_image_files[test_split_index:]
    train_val_files = all_image_files[:test_split_index]
    
    # División 80/20 del grupo restante (Train / Val)
    val_split_index = int(len(train_val_files) * (1 - VAL_RATIO))
    
    val_files = train_val_files[val_split_index:]
    train_files = train_val_files[:val_split_index]
    
    # Comprobación (no deben solaparse)
    assert len(train_files) + len(val_files) + len(test_files) == total_files
    
    print(f"Repartiendo en:")
    print(f"  - Train: {len(train_files)} archivos")
    print(f"  - Val:   {len(val_files)} archivos")
    print(f"  - Test:  {len(test_files)} archivos\n")

```


Copia los archivos a las carpetas correspondientes (entrenamiento, validación, prueba).

```python
    # 4. Copiar los archivos a sus destinos
    copiar_archivos(train_files, "train")
    copiar_archivos(val_files, "val")
    copiar_archivos(test_files, "test")
```


Muestra un mensaje de éxito al finalizar el proceso.

```python
  print("\n¡Reparto completado exitosamente!")
```

## Descripción de VC_P4.ipynb
A continuación, vamos a explicar el código de la práctica implementada, con sus diferentes pasos desglosados.

Entrenamiento del modelo YOLO
```python
import cv2  
import math 
from ultralytics import YOLO
```
- __cv2:__ Se importa la librería OpenCV, que se utiliza comúnmente en visión por computador, aunque no se utiliza en esta parte del código.

- __math:__ Es la librería estándar para operaciones matemáticas, tampoco se utiliza directamente aquí.

- __ultralytics.YOLO:__ Importa la clase YOLO de la librería ultralytics, que es la implementación de YOLOv5 utilizada para entrenar y realizar inferencia con modelos YOLO.

```python
data_path = r"C:\Users\luisp\Desktop\VC\prac1\P4\matriculas\data\matriculas.yaml"
model = YOLO("yolo11s.pt")  # mejor que 11n para pequeños
```
- __data_path:__ Define la ruta del archivo `matriculas.yaml`, que contiene la configuración de los datos de entrenamiento, como las clases de los objetos, la ruta a las imágenes de entrenamiento y validación, y otros parámetros necesarios para el entrenamiento.

- **model = YOLO("yolo11s.pt"):** Carga el modelo preentrenado `yolo11s.pt`. Este es un modelo de la serie `YOLOv5`, entrenado con la arquitectura yolo11s, diseñada para ser más ligera y eficiente en hardware de recursos limitados. La versión `11s` es mejor para trabajar con modelos más pequeños y datasets pequeños.

```python
results = model.train(
    data=data_path,
    epochs=1000,   
    patience=100,
    imgsz=1280,  
    batch=-1,       
    device=0,
    amp=True,
    rect=True,      
    workers=4,
    optimizer="adamw",
    cos_lr=True,
    lr0=8e-4,              
    lrf=1e-2,
    mosaic=0.5,          
    close_mosaic=15,
    mixup=0.0,     
    copy_paste=0.0,
    fliplr=0.0,         
    degrees=3.0,         
    translate=0.05,
    scale=0.70,          
    shear=0.0,
    perspective=0.05,
    hsv_h=0.015, hsv_s=0.4, hsv_v=0.4,  
    save_period=10,
    project="runs/detect",
    name="plates_s_1280_rect_v2",
    exist_ok=True,
)
```
Este bloque es la llamada al método `train` para entrenar el modelo YOLOv5 con los parámetros especificados.

- __data:__ El archivo YAML que contiene la configuración de los datos de entrenamiento.

- __epochs:__ El número de épocas (iteraciones) que el modelo entrenará. Aquí se configura para entrenar durante 1000 épocas.

- __patience:__ Es el número de épocas sin mejora en el rendimiento antes de que el entrenamiento se detenga. Aquí se configura en 100.

- __imgsz:__ El tamaño de las imágenes de entrada (1280x1280 píxeles). Este es un tamaño comúnmente utilizado para obtener un buen equilibrio entre precisión y velocidad.

- __batch:__ Número de imágenes por lote para el entrenamiento. En este caso, se usa el valor -1 para dejar que el tamaño del batch se ajuste automáticamente en función de los recursos de la GPU.

- __device:__ Especifica el dispositivo en el que entrenar (0 corresponde a la primera GPU). Si no tienes GPU, se podría usar cpu.

- __amp:__ Activar la precisión mixta para acelerar el entrenamiento en hardware compatible con la operación de precisión mixta.

- __rect:__ Esta opción asegura que las imágenes no se redimensionen a una forma cuadrada, lo que puede mejorar la eficiencia y los resultados en ciertas arquitecturas.

- __workers:__ Número de hilos para cargar datos durante el entrenamiento. Aquí se usan 4.

- __optimizer:__ El optimizador utilizado, en este caso adamw, que es una versión mejorada del optimizador Adam.

- __cos_lr:__ Si es verdadero, utiliza una programación de la tasa de aprendizaje de tipo coseno, que puede ayudar a mejorar la convergencia.

- __lr0:__ Tasa de aprendizaje inicial.

- __lrf:__ Factor de ajuste para la tasa de aprendizaje durante el entrenamiento.

- __mosaic:__ Porcentaje de mosaico (aumento de datos) que se aplica a las imágenes de entrada para mejorar la generalización.

- __close_mosaic:__ Número de imágenes que se deben usar para aplicar mosaicos. Esto se puede ajustar según el tamaño de la red y los recursos.

- __mixup:__ Técnicas de aumento de datos que combinan diferentes imágenes en un solo conjunto.

- __copy_paste:__ Probabilidad de usar un aumento de datos basado en copiar y pegar partes de una imagen en otra.

- __fliplr:__ Probabilidad de hacer un volteo horizontal aleatorio de las imágenes.

- __degrees:__ Rango de rotación aleatoria de las imágenes (en grados).

- __translate:__ Rango de traslación aleatoria en las imágenes.

- __scale:__ Rango de escalado aleatorio de las imágenes.

- __shear:__ Rango de corte aleatorio de las imágenes.

- __perspective:__ Rango de transformación de perspectiva aleatoria de las imágenes.

- __hsv_h, hsv_s, hsv_v:__ Valores de ajuste para el cambio aleatorio en el espacio de color HSV para mejorar la robustez del modelo.

- __save_period:__ Cada cuántas épocas se guarda el modelo entrenado. Aquí se configura para guardar cada 10 épocas.

- __project:__ Directorio de salida para guardar los resultados del entrenamiento.

- __name:__ El nombre del proyecto (usado para nombrar la carpeta de resultados).

- __exist_ok:__ Si es True, sobrescribe cualquier archivo existente en el directorio de resultados.

Tras el entrenamiento procedemos a ver el resultado del modelo sobre un video y exportación de resultados

```python
import os, csv, cv2
from collections import defaultdict, Counter, deque
import numpy as np
from ultralytics import YOLO
```
- **os, csv, cv2:** Se importan estas librerías para trabajar con archivos, generar archivos CSV, y manipular videos e imágenes.

- **defaultdict, Counter, deque:** Estas clases de la librería collections se usan para almacenar y manejar datos de manera eficiente (como conteos de objetos detectados).

- **numpy:** Se utiliza para realizar operaciones matemáticas y manejar arreglos.

- **ultralytics.YOLO:** Se importa nuevamente la clase YOLO para realizar la inferencia.

```python
VIDEO_IN  = r"C:\Users\luisp\Desktop\VC\prac1\P4\videos\C0142.mp4"
VIDEO_OUT = r"C:\Users\luisp\Desktop\VC\prac1\P4\outputs\video_annotado2.mp4"
CSV_OUT   = r"C:\Users\luisp\Desktop\VC\prac1\P4\outputs\detecciones2.csv"
```
- **VIDEO_IN:** Ruta al archivo de entrada del video que se va a procesar.

- **VIDEO_OUT:** Ruta donde se guardará el video de salida con las anotaciones de detección.

- **CSV_OUT:** Ruta donde se guardarán las detecciones en formato CSV.

Este bloque configura las rutas necesarias para los archivos de entrada y salida.

```python
TARGET_CLASSES = {"person", "car", "motorbike", "bus", "truck"}
TRACKER = "bytetrack.yaml"
DET_CONF = 0.25

PLATE_CONF   = 0.28
PLATE_IOU    = 0.60
PLATE_IMGSZ  = 1280

PLATE_ONLY_BOTTOM_BAND = True
BOTTOM_FRAC   = 0.65
EXTRA_BAND_UP = 0.10

PLATE_AR_MIN, PLATE_AR_MAX = 3.2, 6.6
ALLOW_TWO_LINE = True
MIN_PLATE_AREA = 400

VEH_PLATE_AREA_FRAC_MIN = 0.0015
VEH_PLATE_AREA_FRAC_MAX = 0.08
PLATE_VH_FRAC_MIN = 0.04
PLATE_VH_FRAC_MAX = 0.28

ANONYMIZE = False
USE_CONTOUR_FALLBACK = True
ANTI_HEADLIGHT = True
FLOW_ANALYSIS = True

HIST_N = 7
ACCEPT_K = 3
HOLD_FRAMES = 10

MISSING_TOLERANCE = 12
```
Este bloque de código define varios parámetros utilizados en un sistema de detección y seguimiento de objetos, especialmente centrado en la detección de matrículas de vehículos. Vamos a desglosar qué significa cada parámetro.

- `TARGET_CLASSES`: Es un conjunto de clases de objetos que el modelo YOLO debe detectar. Estas clases son: personas, coches, motocicletas, autobuses y camiones. El sistema se centrará en estas clases al realizar la detección en imágenes o videos.

- `DET_CONF`: Es la confianza mínima para que una detección sea considerada válida. Si el modelo tiene una probabilidad de detección mayor que 0.25, la detección será aceptada. Este parámetro puede ajustarse para filtrar detecciones de baja calidad.

- `PLATE_CONF`: Es la confianza mínima para la detección de matrículas. Si la probabilidad de que una región detectada sea una matrícula es mayor que 0.28, será considerada.

- `PLATE_IOU`: Es el umbral de Intersección sobre Unión (IoU) que define cuándo se considera que dos cajas delimitadoras (bounding boxes) se superponen de forma significativa. Un valor de 0.60 significa que se acepta una superposición del 60% entre dos cajas para ser consideradas una única detección.

- `PLATE_IMGSZ`: Es el tamaño de imagen utilizado para entrenar o procesar las imágenes. Aquí está configurado a 1280x1280 píxeles.

- `PLATE_ONLY_BOTTOM_BAND`: Indica si se debe detectar la matrícula solo en la parte inferior de la imagen. Si está configurado como True, el modelo buscará las matrículas solo en la parte inferior de la imagen.

- `BOTTOM_FRAC`: Define la fracción de la imagen en la que se espera que esté la matrícula. En este caso, se busca en el 65% inferior de la imagen.

- `EXTRA_BAND_UP`: Esta opción permite incluir una pequeña franja adicional por encima de la parte inferior, en este caso un 10% de la altura de la imagen, para capturar matrículas que puedan estar ligeramente más altas.

- `PLATE_AR_MIN y PLATE_AR_MAX`: Estos parámetros definen el rango de la relación de aspecto de las matrículas. El valor debe estar entre 3.2 y 6.6, lo que significa que solo se aceptan cajas delimitadoras cuyo ancho sea al menos 3.2 veces el alto y no más de 6.6 veces el alto.

- `ALLOW_TWO_LINE`: Indica si se permiten matrículas que tengan dos líneas de texto (por ejemplo, si la matrícula tiene más de un conjunto de caracteres en dos filas). Si está configurado en True, se permitirá esta condición.

- `MIN_PLATE_AREA`: Define el área mínima que debe tener una matrícula para ser considerada. Las matrículas demasiado pequeñas (áreas menores a 400 píxeles cuadrados) se ignorarán.

- `VEH_PLATE_AREA_FRAC_MIN y VEH_PLATE_AREA_FRAC_MAX`: Definen el rango de fracciones del área de la matrícula con respecto al área total del vehículo. La matrícula debe ocupar entre un 0.15% y un 8% del área total del vehículo para ser considerada válida.

- `PLATE_VH_FRAC_MIN y PLATE_VH_FRAC_MAX`: Definen el rango de la fracción del área de la matrícula con respecto al área del vehículo. Se requiere que la matrícula ocupe entre el 4% y el 28% del área del vehículo para ser considerada una detección válida.

- `ANONYMIZE`: Si está activado (True), las personas y vehículos pueden ser anonimizados (por ejemplo, blureando rostros o matrículas).

- `USE_CONTOUR_FALLBACK`: Si es True, el sistema utilizará un enfoque de contornos para detectar matrículas si el modelo principal falla.

- `ANTI_HEADLIGHT`: Si está activado, este parámetro ayuda a reducir el impacto de las luces delanteras de los vehículos al realizar la detección.

- `FLOW_ANALYSIS`: Si está activado, se realiza un análisis del flujo de movimiento de los vehículos, lo que puede ayudar a mejorar la detección en escenas con mucho movimiento.

- `HIST_N`: Define el número de fotogramas históricos que se usarán para el análisis de flujo de los vehículos.

- `ACCEPT_K`: Es un umbral que determina cuántas veces un objeto debe ser detectado de forma consistente antes de ser aceptado como una detección válida.

- `HOLD_FRAMES`: Número de fotogramas durante los cuales se mantiene el seguimiento de un objeto. Si el objeto se pierde durante ese número de fotogramas, se abandona el seguimiento.

- `MISSING_TOLERANCE`: Este parámetro define el número máximo de fotogramas que un objeto puede estar ausente (por ejemplo, fuera de la vista o perdido) antes de que se considere que el objeto ya no está presente.

A continuación, vamos a explicar paso a paso lo que hace cada función auxiliar entre todas las que creamos y cómo se integra en el flujo general del proceso.
```python
def clamp_roi(x1, y1, x2, y2, W, H):
    x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
    y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
    return x1, y1, x2, y2
```
- Propósito: Restringe las coordenadas del área de interés (ROI) dentro de los límites de la imagen.

- Uso: Asegura que los valores de las coordenadas de la caja delimitadora no estén fuera del tamaño de la imagen.
```python
def upscale_crop(crop, target_min_side=900, max_side=1600):
    h, w = crop.shape[:2]
    s = max(h, w)
    if s < target_min_side:
        r = target_min_side / s
        nw, nh = int(w*r), int(h*r)
        if max(nw, nh) > max_side:
            r = max_side / max(w, h)
            nw, nh = int(w*r), int(h*r)
        return cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)
    return crop
```
- Propósito: Escala un recorte de la imagen (crop) para que su dimensión más pequeña sea al menos de target_min_side y que no exceda max_side en ninguna dirección.

- Uso: Ajusta el tamaño del recorte de la matrícula para que sea adecuado para la detección.

```python
def plausible_plate_absrel(w, h, veh_w, veh_h):
    area = w * h
    if area < MIN_PLATE_AREA: return False
    ar = w / max(1, h)
    if ALLOW_TWO_LINE and (1.2 <= ar <= 2.2):
        pass
    elif not (PLATE_AR_MIN <= ar <= PLATE_AR_MAX):
        return False
    rel_area = area / (max(1, veh_w) * max(1, veh_h))
    if not (VEH_PLATE_AREA_FRAC_MIN <= rel_area <= VEH_PLATE_AREA_FRAC_MAX):
        return False
    rel_h = h / max(1, veh_h)
    if not (PLATE_VH_FRAC_MIN <= rel_h <= PLATE_VH_FRAC_MAX):
        return False
    return True
```
- Propósito: Verifica si el área y las proporciones de la matrícula son plausibles para un vehículo dado, considerando el área mínima, la relación de aspecto y la fracción del área del vehículo ocupada por la matrícula.

- Uso: Se utiliza para filtrar detecciones de matrículas que no cumplen con las condiciones establecidas (por ejemplo, si son demasiado pequeñas o de proporciones inusuales).

```python
def blur_box(img, x1, y1, x2, y2, k=31):
    x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
    roi = img[y1:y2, x1:x2]
    if roi.size == 0: 
        return img
    blur = cv2.GaussianBlur(roi, (k|1, k|1), 0)
    img[y1:y2, x1:x2] = blur
    return img
```
- Propósito: Aplica un desenfoque gaussiano sobre una región específica de la imagen (es decir, el área de una detección) para anonimizar los datos.

- Uso: Se activa si ANONYMIZE está configurado como True.

```python
def is_headlight_like(crop_box):
    g = cv2.cvtColor(crop_box, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(g, 230, 255, cv2.THRESH_BINARY)[1]
    white_frac = thr.mean() / 255.0
    edges = cv2.Canny(g, 80, 160)
    vert = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    return (white_frac > 0.65 and edges.mean() < 12 and np.abs(vert).mean() < 6)
```

- Propósito: Detecta si una región en la imagen se asemeja a un faro de vehículo (basado en el color y las características de bordes).

- Uso: Se utiliza en el filtro de matrículas para evitar falsos positivos causados por luces brillantes en las imágenes.

```python
def find_plate_by_contours(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0
    Hc, Wc = crop.shape[:2]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 20 or h < 10: 
            continue
        if x < 2 or y < 2 or x+w > Wc-2 or y+h > Hc-2:
            continue
        area = w*h
        if area > best_area:
            best = (x, y, x+w, y+h); best_area = area
    return best
```
- Propósito: Detecta la matrícula en una imagen de recorte usando contornos y detección de bordes.

- Uso: Se utiliza como un método de respaldo para encontrar matrículas si el modelo YOLO no las detecta con alta precisión.

```python
def position_filters(mx1,my1,mx2,my2, vx1,vy1,vx2,vy2):
    veh_w = vx2 - vx1; veh_h = vy2 - vy1
    vx_c = (vx1 + vx2) * 0.5
    mx_c = (mx1 + mx2) * 0.5
    my_c = (my1 + my2) * 0.5
    if abs(mx_c - vx_c) / max(1, veh_w * 0.5) > 0.55:
        return False
    if not (vy1 + 0.45*veh_h <= my_c <= vy1 + 0.95*veh_h):
        return False
    return True
```
- Propósito: Filtra las detecciones de matrículas que no están en la posición correcta dentro de la caja del vehículo.

- Uso: Ayuda a asegurar que las matrículas detectadas están en la ubicación esperada en relación con el vehículo.
```python
def score_plate_abs(px1, py1, px2, py2, pconf, vx1, vy1, vx2, vy2):
    mx_c = (px1+px2)*0.5; my_c = (py1+py2)*0.5
    vx_c = (vx1+vx2)*0.5; vy_c = (vy1+vy2)*0.5
    veh_w = vx2-vx1; veh_h = vy2-vy1
    cx_pen = abs(mx_c - vx_c) / max(1, veh_w*0.5)
    cy_pen = abs(my_c - (vy1+0.78*veh_h)) / max(1, veh_h*0.5)
    return pconf - 0.6*cx_pen - 0.3*cy_pen
```
- Propósito: Calcula una puntuación para la detección de una matrícula basada en la ubicación y la confianza de la predicción.

- Uso: Se utiliza para ajustar las detecciones de matrículas en función de su proximidad al vehículo.

Después de la implementación de estas funciones, pasamos a la captura del vídeo y a realizar las detecciones, así como procesar los resultados obtenidos. Explicaremos paso por paso el procedimiento:

#### 1. Lectura del Video
```python
ok, frame = cap.read()
if not ok:
    break
```
- `cap.read()`: Lee un fotograma del video de entrada. Si no se puede leer el fotograma (por ejemplo, si se ha llegado al final del video), el bucle termina con break.
#### 2. Detección y Seguimiento con YOLO
```python
gen = detector.track(
    source=frame, stream=True, persist=True,
    tracker=TRACKER, conf=DET_CONF, verbose=False
)
```
- `detector.track()`: Utiliza el modelo YOLO para realizar el seguimiento de objetos en el fotograma actual. Devuelve un generador que produce los resultados de la detección y el seguimiento.

- `source=frame`: El fotograma actual del video.

- `stream=True`: Indica que el procesamiento es continuo (en flujo).

- `persist=True`: Mantiene la información entre fotogramas para realizar un seguimiento persistente.

- `tracker=TRACKER`: Utiliza un modelo de seguimiento (definido en TRACKER).

- `conf=DET_CONF`: La confianza mínima de las detecciones (por encima de este valor, las detecciones se consideran válidas).
#### 3. Procesar los Resultados de la Detección
```python
try:
    res = next(gen)
except StopIteration:
    res = None
```
- `next(gen)`: Extrae el siguiente resultado del generador (gen), que contiene las detecciones para el fotograma actual.

- `StopIteration`: Si no hay más resultados (es decir, el generador ha terminado), res se asigna a None.
#### 4. Comprobación de Detecciones
```python
if res is None or res.boxes is None or len(res.boxes) == 0:
    writer.write(frame)
    frame_idx += 1
    continue
```
- Si no se detectan objetos (`res.boxes` es None o está vacío), el fotograma se guarda tal cual en el video de salida (`writer.write(frame)`) y se pasa al siguiente fotograma.
#### 5. Procesar las Detecciones
```python
names = detector.model.names
boxes = res.boxes
active_ids = set()
```
- `names`: Obtiene los nombres de las clases del modelo YOLO (por ejemplo, person, car, etc.).

- `boxes`: Contiene las cajas delimitadoras de las detecciones de objetos.

- `active_ids`: Un conjunto que almacena los identificadores de seguimiento de los objetos detectados.

#### 6. Filtrar y Dibujar Detecciones
```python
for b in boxes:
    if b.cls is None or b.conf is None or b.xyxy is None:
        continue

    cls_id = int(b.cls[0].item())
    conf = float(b.conf[0].item())
    name = names.get(cls_id, str(cls_id))
    if name not in TARGET_CLASSES:
        continue
```
- Para cada detección en `boxes`, se extraen el identificador de clase (`cls_id`), la confianza de la detección (`conf`), y las coordenadas de la caja delimitadora (`xyxy`).

- `TARGET_CLASSES`: Filtra las detecciones para que solo se procesen las clases que nos interesan (por ejemplo, vehículos).

```python
x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
tid = int(b.id[0].item()) if b.id is not None else -1
active_ids.add(tid)
```
- `x1, y1, x2, y2`: Coordenadas de la caja delimitadora (esquina superior izquierda `(x1, y1)` y esquina inferior derecha `(x2, y2)`).

- `tid`: Identificador de seguimiento del objeto. Si no está presente, se asigna -1.
#### 7. Anónimización o Dibujo de las Detecciones
```python
if ANONYMIZE:
    frame = blur_box(frame, x1, y1, x2, y2, k=35)
else:
    color = (0, 255, 0) if name != "person" else (0, 200, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"{name} {conf:.2f} ID:{tid}",
                (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
```
- __Anónimización__: Si `ANONYMIZE` está activado, se aplica un desenfoque a la caja delimitadora.

- __Dibujo de la detección__: Si no se anonimiza, se dibuja un rectángulo alrededor del objeto y se muestra su clase, confianza y el ID de seguimiento.

#### 8. Detección de Matrículas en Vehículos
```python
plate_flag, plate_conf, (mx1,my1,mx2,my2), plate_text = 0, 0.0, (0,0,0,0), ""
if name in {"car","motorbike","bus","truck"}:
    # Cálculos para obtener la región donde se encuentra la matrícula
```
- Si la clase detectada es un vehículo (coche, motocicleta, autobús, camión), se realiza una serie de cálculos para determinar la región de la imagen que probablemente contenga la matrícula.
```python
crop = frame[ry1:ry2, rx1:rx2]
crop_up = upscale_crop(crop, target_min_side=900, max_side=1600)
```
- `crop`: Recorta la región del vehículo para centrarse en la zona que podría contener la matrícula.

- `crop_up`: Se escala la imagen recortada para mejorar la calidad de la detección de la matrícula.
```python
pp = plate_model.predict(
    source=crop_up,
    conf=PLATE_CONF,
    iou=PLATE_IOU,
    imgsz=PLATE_IMGSZ,
    max_det=5,
    augment=True,
    agnostic_nms=False,
    verbose=False
)
```
- `plate_model.predict()`: Utiliza el modelo de detección de matrículas para predecir la ubicación de las matrículas en el recorte escalado.

#### 9. Filtrado de la Mejor Detección de Matrícula
```python
if pp and len(pp[0].boxes) > 0:
    for pb in pp[0].boxes:
        px1, py1, px2, py2 = pb.xyxy[0].tolist()
        pconf = float(pb.conf[0].item())
        ...
```
- `pp[0].boxes`: Accede a las cajas delimitadoras predichas para las matrículas en el recorte.

- Se filtran las detecciones que no cumplen con los criterios de tamaño, relación de aspecto y otros parámetros, como la posición.
#### 10. Almacenamiento de Detecciones de Matrículas y Vehículos
```python
push_history(tid_key, (bx1,by1,bx2,by2,bconf))
```
- `push_history`: Almacena las detecciones en un historial para realizar un seguimiento de las matrículas a lo largo de los fotogramas.

#### 11. Anotación en el Video
```python
if g:
    gx1, gy1, gx2, gy2, gc = g
    plate_flag, plate_conf = 1, gc
    mx1,my1,mx2,my2 = gx1,gy1,gx2,gy2
    cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
    cv2.putText(frame, f"PLATE {gc:.2f}", (gx1, max(0, gy1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
```
- Si se encuentra una matrícula válida, se dibuja un rectángulo alrededor de ella y se muestra la confianza de la detección en el video.

#### 12. Escribir Resultados en el CSV
```python
cw.writerow([frame_idx, name, f"{conf:.3f}", tid, x1, y1, x2, y2,
            plate_flag, f"{plate_conf:.3f}", mx1, my1, mx2, my2, ""])
```
- Escribe los resultados de la detección (nombre del objeto, confianza, ID, coordenadas de la caja delimitadora, etc.) en un archivo CSV.
#### 13. Análisis del Flujo 
```python
if FLOW_ANALYSIS:
    ids_to_check = set(list(last_centroid.keys()) + list(inactive_counter.keys()))
    for gid in list(ids_to_check):
        ...
```
- `FLOW_ANALYSIS`: Si está activado, realiza un análisis del flujo de objetos (por ejemplo, cuántos objetos salen de la vista por los bordes de la imagen).
#### 14. Escribir el Video de Salida
```python
writer.write(frame)
frame_idx += 1
```
- `writer.write(frame)`: Escribe el fotograma procesado (con las anotaciones) en el video de salida.

**Resumen**

El bloque de código realizado se encarga de leer cada fotograma del video, realizar la detección de objetos con YOLO, rastrear los vehículos y matrículas, y escribir los resultados tanto en un archivo CSV como en un nuevo video con anotaciones. Además, realiza análisis de flujo de objetos y opciones de anonimización de las matrículas si se requieren.

---



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

