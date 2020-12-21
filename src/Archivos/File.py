import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

class Imagen:

    def __init__(self, imagen, salida):
        self.imagen = imagen
        self.salida = salida


# ruta   : carpeta donde se obtiene los archivos de entrenamiento 
# modelo : tipo de modelo a generar por universixades
# modelo => 1 : Landivar , 2 : Mariano, 3 : Marroquin, 4 : USAC  
def cargarImagenes(ruta, modelo):
    uni = 0
    if   modelo == 1:
        uni = "Landivar"
    elif modelo == 2:
        uni = "Mariano"
    elif modelo == 3:
        uni = "Marroquin"
    elif modelo == 4:
        uni = "USAC"

    result_entradas = np.array([])
    # Obtener entradas y salidas 1 
    with os.scandir(ruta + uni) as ficheros:
        for fichero in ficheros:
            img = cv2.imread(fichero.path, 1)
            result_entradas = np.append(result_entradas, Imagen(img, 1))
    
    # Obtener entradas y salidas 0
    with os.scandir(ruta) as ficheros:
        for fichero in ficheros:
            carpeta = fichero.name
            if carpeta != uni:
                with os.scandir(ruta + carpeta) as imagenes:
                    for archivo in imagenes:
                        img = cv2.imread(archivo.path, 1)
                        result_entradas = np.append(result_entradas, Imagen(img, 0))
    result = np.array(result_entradas)
    np.random.shuffle(result)
    result_entradas = [o.imagen for o in result]
    result_salidas  = [o.salida for o in result]
    result_entradas = np.array(result_entradas)
    result_salidas  = np.array(result_salidas)

    cant_entradas = int(len(result_entradas) * 80 / 100)
    train_set = result_entradas[0 : cant_entradas, :]
    test_set  = result_entradas[cant_entradas:, :]

    # Se separan las entradas de las salidas
    train_set_x_orig = train_set
    train_set_y_orig = np.array([result_salidas[: cant_entradas]])

    test_set_x_orig = test_set
    test_set_y_orig = np.array([result_salidas[cant_entradas:]])

    #imagenes_usac = np.array(imagenes_usac)
    #cantidad_entrada = int(80 * len(imagenes_usac) / 100)
    #cantidad_salida  = len(imagenes_usac) - cantidad_entrada
    #entradas_usac = np.array(imagenes_usac[:cantidad_entrada][:])
    #salidas_usac  = np.array(imagenes_usac[-cantidad_salida:][:])
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, ['No ' + uni, uni]

#cargarImagenes("../Datasets/", 4)

"""
my_img = cv2.imread('Untitled.png') 
inverted_img = (255.0 - my_img)  
final = inverted_img / 255.0

# Visualize the result
plt.imshow(final)
plt.show()

print(final.shape)
(661, 667, 3)"""