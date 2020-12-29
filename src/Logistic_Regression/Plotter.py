from Logistic_Regression.Model import Model
import matplotlib.pyplot as chart


def show_picture(pixels):
    chart.imshow(pixels)
    chart.show()


def show_Model(models, ruta_imagen, ylim1, ylim2, xlim1 = 0, xlim2 = 1000):
    for model in models:
        chart.plot(model.bitacora, label=str(model.alpha))
    chart.ylabel('Costo')
    chart.xlabel('Iteraciones')
    legend = chart.legend(loc='upper center', shadow=True)
    chart.ylim(ylim1, ylim2)
    chart.xlim(xlim1, xlim2)
    chart.savefig(ruta_imagen + ".png")
    chart.show()
    
