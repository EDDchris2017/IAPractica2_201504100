from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_session import Session
import json
import os

app = Flask(__name__)
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)


@app.route('/status')
def status():
    return "Funcionando servidor Practica 2 IA"

@app.route('/imagenes', methods=['GET', 'POST'])
def imagenes():
    files = request.files.getlist('files[]')
    file_names = []
    for file in files:
        file_names.append(file.filename)
        file.save("../img_usuario/" + file.filename)
    return render_template('index.html', filenames=file_names)

@app.route('/display/<filename>')
def display_image(filename):
	print('display_image filename: ' + filename)
	return redirect(url_for("../img_usuario/", filename='uploads/' + filename), code=301)

@app.route('/')
def home_form():
    return render_template("index.html", modelo= 0, res = "")


if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000, debug=True)
