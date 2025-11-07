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

## Descripción de la Práctica 4
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


```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```


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

### Tarea:
