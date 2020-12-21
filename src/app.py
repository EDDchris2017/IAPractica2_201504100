from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import json

app = Flask(__name__)
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)


@app.route('/status')
def status():
    return "Funcionando servidor Flask Practica 2 IA"

@app.route('/')
def home_form():
    return "Aplicacion Practica 2"


if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000, debug=True)
