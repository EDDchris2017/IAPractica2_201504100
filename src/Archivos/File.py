import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

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

    result_entradas = []
    # Obtener entradas y salidas 1 
    with os.scandir(ruta + uni) as ficheros:
        for fichero in ficheros:
            img = cv2.imread(fichero.path, 1)
            result_entradas.append([img, 1])
    
    # Obtener entradas y salidas 0
    with os.scandir(ruta) as ficheros:
        for fichero in ficheros:
            carpeta = fichero.name
            if carpeta != uni:
                with os.scandir(ruta + carpeta) as imagenes:
                    for archivo in imagenes:
                        img = cv2.imread(fichero.path, 1)
                        result_entradas.append([img, 0])
    result_entradas = np.array(result_entradas)
    np.random.shuffle(result_entradas)
    #result_salidas  = result_entradas[:,1:]
    cant_entradas = int(len(result_entradas) * 80 / 100)
    train_set = result_entradas[0 : cant_entradas, :]
    test_set  = result_entradas[cant_entradas:, :]
    
    # Se separan las entradas de las salidas
    train_set_x_orig = train_set[:, 0:]
    train_set_y_orig = train_set[:, 1:]

    test_set_x_orig = test_set[:, 0:]
    test_set_y_orig = test_set[:, 1:]

    #imagenes_usac = np.array(imagenes_usac)
    #cantidad_entrada = int(80 * len(imagenes_usac) / 100)
    #cantidad_salida  = len(imagenes_usac) - cantidad_entrada
    #entradas_usac = np.array(imagenes_usac[:cantidad_entrada][:])
    #salidas_usac  = np.array(imagenes_usac[-cantidad_salida:][:])

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, ['No ' + uni, uni]

cargarImagenes("../Datasets/", 4)

"""
my_img = cv2.imread('Untitled.png') 
inverted_img = (255.0 - my_img)  
final = inverted_img / 255.0

# Visualize the result
plt.imshow(final)
plt.show()

print(final.shape)
(661, 667, 3)"""