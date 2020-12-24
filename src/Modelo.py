#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Archivos import File
from Logistic_Regression.Model import Model
from Logistic_Regression.Data import Data
from Logistic_Regression import Plotter
import numpy as np
import pickle

ONLY_SHOW = False #Veo si quiero mostrar una imagen del conjunto de datos

# modelo => 1 : Landivar , 2 : Mariano, 3 : Marroquin, 4 : USAC  
def entrenarModelos(uni):
    id_archivo = ""
    if uni == 4:
        id_archivo = "USAC"
    elif uni == 3:
        id_archivo = "Marroquin"
    elif uni == 2:
        id_archivo = "Mariano"
    elif uni == 1:
        id_archivo = "Landivar"
    name = "Modelo_" + id_archivo
    # Cargando conjunto de datos
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = File.cargarImagenes("Datasets/", uni)

    # Convertir imagenes a un solo arreglo
    train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    #Vemos cómo queda ahora la estructura de una imagen
    #(12288, 209) En este caso tiene 209 registros y cada registro tiene 12288 valores
    #En el caso de las notas cada registro tenía solo 3 valores, que eran las 3 notas
    #Por lo tanto, nuestro modelo va a tener 12288 + 1 Coeficientes, el + 1 es por B0
    #print(train_set_x.shape)

    # Vean la diferencia de la conversion
    print('Original: ', train_set_x_orig.shape)
    print('Con reshape: ', train_set_x.shape)

    #print('tamaño train_set_x_orig: ', len(train_set_x_orig))
    #print('tamaño train_set_x: ', len(train_set_x))

    #temp = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1)
    #print('Prueba: ', temp.shape)

    #print('train_set_x')
    #print(train_set_x)
    #exit()


    # Definir los conjuntos de datos
    train_set = Data(train_set_x, train_set_y, 255)
    test_set = Data(test_set_x, test_set_y, 255)

    # Se entrenan los modelos
    #model1 = Model(train_set, test_set, reg=False, alpha=0.0001, lam=0)
    #model1.training()

    #model2 = Model(train_set, test_set, reg=True, alpha=0.01, lam=150) #Aquí también se puede ver sobre-ajuste
    #model2 = Model(train_set, test_set, reg=True, alpha=0.01, lam=1) #Se puede ver en la gráfica que hay SOBRE-AJUSTE

    #model2 = Model(train_set, test_set, reg=True, alpha=0.001, lam=300) #Se ajusta mejor con la regulariación de 300, pero se tarda más
    
    # Modelo 1
    model1 = Model(train_set, test_set, reg=False, alpha=0.001, lam=150, MAX_ITERATIONS=10000, MIN_VALUE=0.0, STEP=10) #Baja más quitandole la regularización
    model1.training()
    guardarBitacora(model1, id_archivo, 1)
    guardarModelo(model1, id_archivo, 1)

    # Modelo 2
    model2 = Model(train_set, test_set, reg=True, alpha=0.00001, lam=20, MAX_ITERATIONS=100000, MIN_VALUE=0.3, STEP=50)
    model2.training()
    guardarBitacora(model2, id_archivo, 2)
    guardarModelo(model1, id_archivo, 2)

    # Modelo 3
    model3 = Model(train_set, test_set, reg=False, alpha=0.000001, lam=100, MAX_ITERATIONS=100000, MIN_VALUE=0.5, STEP=100)
    model3.training()
    guardarBitacora(model3, id_archivo, 3)
    guardarModelo(model1, id_archivo, 3)

    # Modelo 4
    model4 = Model(train_set, test_set, reg=False, alpha=0.001, lam=10, MAX_ITERATIONS=100000, MIN_VALUE=0.0, STEP=200)
    model4.training()
    guardarBitacora(model4, id_archivo, 4)
    guardarModelo(model1, id_archivo, 4)

    # Modelo 5
    model5 = Model(train_set, test_set, reg=True, alpha=0.001, lam=50, MAX_ITERATIONS=100000, MIN_VALUE=0.0, STEP= 5)
    model5.training()
    guardarBitacora(model5, id_archivo, 5)
    guardarModelo(model1, id_archivo, 5)

    Plotter.show_Model([model1, model2, model3, model4, model5], "imagen_" + id_archivo)

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

class Objeto:
    def __init__(self, atrib1, atrib2):
        self.atrib1 = atrib1
        self.atrib2 = atrib2
    def display(self):
        print("ATRIB 1 ", self.atrib1, " ATRIB 2: ", self.atrib2)

def pruebaPickle(id_archivo, id):
    me = Objeto("Christopher", 34)
    ruta = "../modelo" + id_archivo + "/modelo_" + str(id) + ".ml"
    print(ruta)
    pickle.dump(me, open(ruta, "wb"))

entrenarModelos(4)