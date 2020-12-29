#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Archivos import File
from Logistic_Regression.Model import Model
from Logistic_Regression.Data import Data
from Logistic_Regression import Plotter
import numpy as np
import os
import pickle

ONLY_SHOW = False #Veo si quiero mostrar una imagen del conjunto de datos

# modelo => 1 : Landivar , 2 : Mariano, 3 : Marroquin, 4 : USAC  
def entrenarModelos(uni, id,reg, alpha, lam, MAX_ITERATIONS, MIN_VALUE, STEP  ):
    id_archivo = ""
    if uni == 4:
        id_archivo = "USAC"
    elif uni == 3:
        id_archivo = "Marroquin"
    elif uni == 2:
        id_archivo = "Mariano"
    elif uni == 1:
        id_archivo = "Landivar"
    # Cargando conjunto de datos
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = File.cargarImagenes("Datasets/", uni)

    # Convertir imagenes a un solo arreglo
    train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    # Vean la diferencia de la conversion
    print('Original: ', train_set_x_orig.shape)
    print('Con reshape: ', train_set_x.shape)


    # Definir los conjuntos de datos
    train_set = Data(train_set_x, train_set_y, 255)
    test_set = Data(test_set_x, test_set_y, 255)

    # Modelo 1
    model1 = Model(train_set, test_set, reg, alpha, lam, MAX_ITERATIONS, MIN_VALUE, STEP) 
    model1.training()
    guardarBitacora(model1, id_archivo, id)
    guardarModelo(model1, id_archivo, id)

    Plotter.show_Model([model1], "imagen_" + id_archivo, ylim1 = 0, ylim2 = 20)

# modelo => 1 : Landivar , 2 : Mariano, 3 : Marroquin, 4 : USAC  
def guardarBitacora(modelo, id_archivo, id):
    name = "Modelo_" + id_archivo
    cad_registro = "------" + "Modelo " + str(id) + "------" + "\n"
    cad_registro += "Entrenamiento = " + str(modelo.train_accuracy) + "\n"
    cad_registro += "Validacion = " + str(modelo.test_accuracy) + "\n"
    cad_registro += "Alpha      = " + str(modelo.alpha) + "\n"
    cad_registro += "Lam        = " + str(modelo.lam) + "\n"
    cad_registro += "Regulacion = " + str(modelo.reg) + "\n"
    cad_registro += "MAX ITERATIONS = " + str(modelo.MAX_ITERATIONS) + "\n"
    cad_registro += "MIN_VALUE      = " + str(modelo.MIN_VALUE) + "\n"
    cad_registro += "STEP           = " + str(modelo.STEP) + "\n"

    f = open(name + ".txt", 'a')
    f.write('\n' + cad_registro + '\n')
    f.close()

def guardarModelo(modelo, id_archivo, id ):
    ruta = "../modelo" + id_archivo + "/modelo_" + str(id) + ".ml"
    pickle.dump(modelo, open(ruta, "wb"))

def abrirModelo(ruta):
    mod = pickle.load(open(ruta, "rb"))
    return mod


def guardarGrafica(uni, ylim1, ylim2, xlim1 = 0, xlim2 = 1000):
    id_archivo = ""
    if uni == 4:
        id_archivo = "USAC"
    elif uni == 3:
        id_archivo = "Marroquin"
    elif uni == 2:
        id_archivo = "Mariano"
    elif uni == 1:
        id_archivo = "Landivar"
    carpeta = "../modelo" + id_archivo
    modelos = []
    with os.scandir(carpeta) as ficheros:
        for fichero in ficheros:
            modelos.append(abrirModelo(fichero.path))
    # Graficar todos los modelos
    Plotter.show_Model(modelos, "imagen_" + id_archivo, ylim1, ylim2, xlim1, xlim2)



#Entrenamiento USAC
#entrenarModelos(4, 1, reg=False, alpha=0.001, lam=150, MAX_ITERATIONS=10000, MIN_VALUE=0.0, STEP=10)
#entrenarModelos(4, 2, reg=True, alpha=0.00001, lam=20, MAX_ITERATIONS=10000, MIN_VALUE=0.3, STEP=10)
#entrenarModelos(4, 3, reg=False, alpha=0.0001, lam=20, MAX_ITERATIONS=15000, MIN_VALUE=0.1, STEP=30)
#entrenarModelos(4, 4, reg=False, alpha=0.0009, lam=20, MAX_ITERATIONS=16000, MIN_VALUE=0.2, STEP=10)
#entrenarModelos(4, 5, reg=True, alpha=0.009, lam=3, MAX_ITERATIONS=15000, MIN_VALUE=0.4, STEP=30)
#guardarGrafica(4, ylim1 = 0, ylim2 = 1.50)

#Entrenamiento Marroquin
#entrenarModelos(3, 1, reg=False, alpha=0.0003, lam=150, MAX_ITERATIONS=12000, MIN_VALUE=0.1, STEP=20)
#entrenarModelos(3, 2, reg=False, alpha=0.0007, lam=150, MAX_ITERATIONS=12000, MIN_VALUE=0.0, STEP=10)
#entrenarModelos(3, 3, reg=True, alpha=0.005, lam=200, MAX_ITERATIONS=10000, MIN_VALUE=0.0, STEP=30)
#entrenarModelos(3, 4, reg=True, alpha=0.005, lam=100, MAX_ITERATIONS=1500, MIN_VALUE=0.2, STEP=10)
#entrenarModelos(3, 5, reg=True, alpha=0.002, lam=600, MAX_ITERATIONS=15000, MIN_VALUE=0.0, STEP=10)
#guardarGrafica(3, ylim1 = 0, ylim2 = 1.50, xlim1=0, xlim2 = 30)

#Entrenamiento Mariano
#entrenarModelos(2, 1, reg=True, alpha=0.004, lam=110, MAX_ITERATIONS=12000,  MIN_VALUE=0.1, STEP=10)
#entrenarModelos(2, 2, reg=True, alpha=0.002, lam=230, MAX_ITERATIONS=12000,  MIN_VALUE=0.0, STEP=10)
#entrenarModelos(2, 3, reg=True, alpha=0.0001, lam=0.5, MAX_ITERATIONS=12000, MIN_VALUE=0.0, STEP=10)
#entrenarModelos(2, 4,  reg=True, alpha=0.0006, lam=130, MAX_ITERATIONS=15000,  MIN_VALUE=0.0, STEP=10)
#entrenarModelos(2, 5,  reg=True, alpha=0.0000009, lam=125, MAX_ITERATIONS=10000,  MIN_VALUE=0.0, STEP=10)
#guardarGrafica(2, ylim1 = 0, ylim2 = 1.50, xlim1=0, xlim2 = 300)

#Entrenamiento Landivar
#entrenarModelos(1, 1,  reg=False, alpha=0.00009, lam=125, MAX_ITERATIONS=10000,  MIN_VALUE=0.1, STEP=15)
#entrenarModelos(1, 2,  reg=False, alpha=0.0004, lam=125, MAX_ITERATIONS=10000,  MIN_VALUE=0.2, STEP=10)
#entrenarModelos(1, 3,  reg=True, alpha=0.0001, lam=200, MAX_ITERATIONS=15000,  MIN_VALUE=0.0, STEP=10)
#entrenarModelos(1, 4,  reg=True, alpha=0.001, lam=150, MAX_ITERATIONS=10000,  MIN_VALUE=0.0, STEP=10)
#entrenarModelos(1, 5, reg=True, alpha=0.002, lam=230, MAX_ITERATIONS=20000,  MIN_VALUE=0.0, STEP=10)
guardarGrafica(1, ylim1 = 0, ylim2 = 1.50, xlim1=0, xlim2 = 100)
