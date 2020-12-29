import pickle
import os
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
            carpeta = "static/uploads"
            # recorrer imagenes generadas
            with os.scandir(carpeta) as ficheros:
                pass
        else:
            print("Hay modelos que no se encuentran cargados ")
        pass
    
    def hayModelos(self):
        return self.modelo_usac != None and self.modelo_landivar != None and self.modelo_mariano != None and self.modelo_marro != None
