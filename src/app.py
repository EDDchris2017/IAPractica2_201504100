from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_session import Session
import json
import shutil
import os
from Predecir import *

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)

# Carga de Modelos de la usac
predecir = Predecir(3,2,4,2)
predecir.abrirModelos()

def borrarCarpeta():
    folder = "static/uploads"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

@app.route('/status')
def status():
    return "Funcionando servidor Practica 2 IA"

@app.route('/imagenes', methods=['GET', 'POST'])
def imagenes():
    borrarCarpeta()
    files = request.files.getlist('files[]')
    file_names = []
    for file in files:
        file_names.append(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    return render_template('index.html', filenames=file_names)
    

@app.route('/display/<filename>')
def display_image(filename):
	print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/analizar', methods=['GET', 'POST'])
def calcular():
    # Ejecutar Predeccion de archivos cargados
    predicciones = predecir.ejecutar()
    if len(predicciones) < 6:
        return render_template('index.html', predicciones=predicciones)
    else:
        return "Mas de 5 imagenes"

    return "Retorno invalido"


@app.route('/')
def home_form():
    return render_template("index.html", modelo= 0, res = "")


if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000, debug=True)
