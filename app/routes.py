from flask import render_template, request
import numpy as np
from .model import GalaxyClassifier

classifier = GalaxyClassifier()

def init_routes(app):
    @app.route('/')
    def home():
        return render_template('index.html')
    
    @app.route('/predict', methods=['POST'])
    def predict():
        features = [
            float(request.form['u']),
            float(request.form['g']),
            float(request.form['r']),
            float(request.form['i']),
            float(request.form['z']),
            float(request.form['u']) - float(request.form['g']),
            float(request.form['g']) - float(request.form['r'])
        ]
        return render_template('index.html', results=classifier.predict(features))