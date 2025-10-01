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


# Práctica 2 de la asignatura Visión por computador.

<details>
<summary><b>📚 Tabla de contenidos</b></summary>


</details>

 ## Instalación de dependencias
 
 Para poder realizar esta práctica, vamos a utilizar las dependencias instaladas anteriormente en el kernel, mas PIL:
 
 - cv2 
 - numpy
 - matplotlib
 - Pil

 ## Tarea 1: Realiza la cuenta de píxeles blancos por filas (en lugar de por columnas). Determina el valor máximo de píxeles blancos para filas, maxfil, mostrando el número de filas y sus respectivas posiciones, con un número de píxeles blancos mayor o igual que 0.90*maxfil.

 El códgo desarrollado fue el siguiente:


 ```python
img = cv2.imread('mandril.jpg') 

canny = cv2.Canny(img, 100, 200)
#El contenido de la imagen resultado de Canny, son valores 0 o 255, lo compruebas al descomentar
#print(canny)
#Cuenta el número de píxeles blancos (255) por columna
#Suma los valores de los pixeles por columna
col_counts = cv2.reduce(canny, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)
row_counts = cv2.reduce(canny, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)

#Normaliza en base al número de filas, primer valor devuelto por shape, y al valor máximo del píxel (255)
#El resultado será el número de píxeles blancos por columna
rows = col_counts[:,] / (255 * canny.shape[0])
cols = row_counts[:,] / (255 * canny.shape[1])

#Muestra dicha cuenta gráficamente
plt.figure()
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Canny")
plt.imshow(canny, cmap='gray') 

plt.subplot(1, 2, 2)
plt.title("Respuesta de Canny")
plt.xlabel("Columnas")
plt.ylabel("% píxeles")
plt.plot(cols)
#Rango en x definido por las columnas
plt.xlim([0, canny.shape[0]])
 ```

 En este caso lo primero que hacemos es leer una imagen, calcular los bordes gracias a Canny , contar cuantos píxeles ha por columnas y por filas, normalizamos para expresarlo en porcentaje y muestra la imagen procesada por canny.

 Ahora pasaremos a desglosarlo por partes, nos saltaremos la lectura de la imagen en disco, y todo lo mencionado en la anterior práctica.

 ```python
canny = cv2.Canny(img, 100, 200)
 ```

 Aplica el detector Canny. Su umbral inferior será de 100 y su umbral superior de 200 y dará una imagen binaria con los valores 0 para negro y los valores 255 para blanco.

 ```python
row_counts = cv2.reduce(canny, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)
 ```
 No vamos a explicar el `col_counts`debido a que, se explica en la propia práctica, sin embargo, en este caso, estamos contando las filas en vez de las columnas marcando el segundo valor de la cuentade Canny y sumando cada fila.

El resto del ejercicio consiste en la visualizació de lo calculado usando matplot lib como en en caso anterior.

## Tarea 2: TAREA: Aplica umbralizado a la imagen resultante de Sobel (convertida a 8 bits), y posteriormente realiza el conteo por filas y columnas similar al realizado en el ejemplo con la salida de Canny de píxeles no nulos. Calcula el valor máximo de la cuenta por filas y columnas, y determina las filas y columnas por encima del 0.90*máximo. Remarca con alguna primitiva gráfica dichas filas y columnas sobre la imagen del mandril. ¿Cómo se comparan los resultados obtenidos a partir de Sobel y Canny?
```python
# Aplica Sobel y umbralizado
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ggris = cv2.GaussianBlur(gris, (3, 3), 0)
sobelx = cv2.Sobel(ggris, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(ggris, cv2.CV_64F, 0, 1)
sobel = cv2.add(sobelx, sobely)
sobel_8u = cv2.convertScaleAbs(sobel)
_, sobel_bin = cv2.threshold(sobel_8u, 50, 255, cv2.THRESH_BINARY)

# Conteo por filas y columnas
row_counts_sobel = np.count_nonzero(sobel_bin, axis=1)
col_counts_sobel = np.count_nonzero(sobel_bin, axis=0)

# Máximos y umbral del 90%
max_rows = np.max(row_counts_sobel)
max_cols = np.max(col_counts_sobel)
umbral_rows = 0.98 * max_rows
umbral_cols = 0.98 * max_cols

# Filas y columnas destacadas
filas_destacadas = np.where(row_counts_sobel >= umbral_rows)[0]
columnas_destacadas = np.where(col_counts_sobel >= umbral_cols)[0]

# Remarca filas y columnas sobre la imagen original
img_marcada = img.copy()
for row in columnas_destacadas:
    cv2.line(img_marcada, (0, row), (img.shape[1]-1, row), (0,255,0), 1)
for col in filas_destacadas:
    cv2.line(img_marcada, (col, 0), (col, img.shape[0]-1), (0,0,255), 1)

print(f"Filas destacadas: {filas_destacadas}")
# Visualización
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title("Sobel binarizado")
plt.axis("off")
plt.imshow(sobel_bin, cmap='gray')
plt.subplot(1,2,2)
plt.title("Mandril con filas/columnas destacadas")
plt.axis("off")
plt.imshow(cv2.cvtColor(img_marcada, cv2.COLOR_BGR2RGB))
plt.show()

# Comparación con Canny
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Conteo columnas (Canny)")
plt.plot(cols)
plt.subplot(1,2,2)
plt.title("Conteo columnas (Sobel)")
plt.plot(row_counts_sobel)
plt.show()
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
- **Tarea 4**: `Guillermo` 
- **README**: `Luis`  

--- 