# -*- coding: utf-8 -*-
"""
Created on Wed May 11 08:09:22 2022

@author: 
"""

# flask app
# importing libraries
import numpy as np
from collections.abc import Mapping
from flask import Flask, request, jsonify, render_template
import pickle

from markupsafe import escape
# flask app
app = Flask(__name__)
# loading model
model = pickle.load(open('model.pkl', 'rb'))
#s
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/prediction' ,methods = ['POST'])
def prediction():
    final_features = [float(x) for x in request.form.values()]
    final_features = [np.array(final_features)]
    prediction = model.predict(final_features)

    return render_template('index.html', output='Reached on time or not {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)