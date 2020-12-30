import pickle
import os
import cv2
import numpy as np
from Logistic_Regression.Data import Data
from PIL import Image
class Prediccion:

    def __init__(self, imagen, usac, marroquin, mariano, landivar):
        self.imagen    = imagen.name
        self.usac      = usac[0]
        self.marroquin = marroquin[0]
        self.mariano   = mariano[0]
        self.landivar  = landivar[0]
        self.principal = ""
        if usac == 1:
            self.principal = "USAC"
        elif marroquin == 1:
            self.principal = "Marroquin"
        elif mariano == 1:
            self.principal = "Mariano"
        elif landivar == 1:
            self.principal = "Landivar"

class Predecir:

    def __init__(self, usac, marroquin, mariano, landivar):
        self.usac       = usac
        self.marroquin  = marroquin
        self.mariano    = mariano
        self.landivar   = landivar
        self.modelo_usac    = None
        self.modelo_marro   = None
        self.modelo_mariano = None
        self.modelo_landivar= None

    # Abrir modelos
    def abrirModelos(self):
        try:
            self.modelo_usac = pickle.load(open("../modeloUSAC/modelo_" + str(self.usac) + ".ml", "rb"))
            self.modelo_marro = pickle.load(open("../modeloMarroquin/modelo_" + str(self.marroquin) + ".ml", "rb"))
            self.modelo_mariano = pickle.load(open("../modeloMariano/modelo_" + str(self.mariano) + ".ml", "rb"))
            self.modelo_landivar = pickle.load(open("../modeloLandivar/modelo_" + str(self.landivar) + ".ml", "rb"))
            print("===> Modelos Cargados")
        except:
            print("===> Error al cargar modelos")
    
    def ejecutar(self):
        if self.hayModelos():
            carpeta = "static/uploads/"
            # recorrer imagenes generadas
            salida = []
            with os.scandir(carpeta) as ficheros:
                # TODO Configurar lectura de archivos
                archivos = list(ficheros)
                if len(archivos) < 6:
                    for imagen in archivos:
                        result = self.predecirArchivo(imagen)
                        salida.append(result)
                    return salida
        else:
            print("Hay modelos que no se encuentran cargados !!!")
            return None
    
    #Entre 1 a 5 archivos
    def predecirArchivo(self, imagen, salida = 0):
        data_imagen = self.imagenVector(imagen, salida)
        # Evaluar con Modelo
        #data_imagen = Data(analizar,np.array(salida),255)
        res_usac        = self.modelo_usac.predict(data_imagen.x)
        res_mariano     = self.modelo_mariano.predict(data_imagen.x)
        res_marroquin   = self.modelo_marro.predict(data_imagen.x)
        res_landivar    = self.modelo_landivar.predict(data_imagen.x)
        return Prediccion(imagen = imagen, usac=res_usac, marroquin=res_marroquin, mariano=res_mariano, landivar=res_landivar)

    def imagenVector(self, imagen, salida):
        img = cv2.imread(imagen.path, 1)
        entradas = np.array([img])
        salidas  = np.array([salida])

        salidas  = salidas.reshape( (1, salidas.shape[0]))

        train_entrada = entradas.reshape( entradas.shape[0], -1).T
        data_set = Data(train_entrada, salidas, 255)

        return data_set

        #img_reshape = entradas.reshape(entradas.shape[0], -1).T
        #y = sum(img_reshape.tolist(),[])
        #x = np.array([y]).T
        #print(x.shape)
        #print(x)
        #return x

    
    def hayModelos(self):
        return self.modelo_usac != None and self.modelo_landivar != None and self.modelo_mariano != None and self.modelo_marro != None

#predecir = Predecir(3,2,4,2)
#predecir.abrirModelos()

#predecir.ejecutar()