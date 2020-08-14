# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 13:54:15 2020

@author: Harun
"""
import pickle
from flask import Flask, request, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)

def f3(array):
    if array[0] >= 18:
        val = 1
    else:
        val = 0
    return val
    
def f5(array):
    if array[0] >= 18 and array[1] == 0 and array[4] == 1:
        val = 3 # Mrs
    elif array[0] < 18 and array[1] == 1:
        val = 0 # Master
    elif array[0] >= 18 and array[1] == 1:
        val = 2 # Mr
    elif array[0] < 18 and array[1] == 0:
        val = 1 # Miss
    else:
        val = 4
    return val

def f6(array):
    for x in array:
        fsize = array[3] + array[4] + 1
    if fsize == 1:
        val = 0 # single
    elif fsize > 1 and fsize < 5:
        val = 1 # small
    elif fsize > 4:
        val = 2
    return val

@app.route('/')
def home():
    return render_template('index.html')
'''
'''



def ValuePredictor(to_predict_list): 
    to_predict = np.array(to_predict_list).reshape(1, 6)
    arraydf = pd.DataFrame(to_predict).copy()
    arraydf[6] = arraydf.apply(f3, axis=1)
    arraydf[7] = arraydf.apply(f5, axis=1)
    arraydf[8] = arraydf.apply(f6, axis=1)
    to_predict = np.array(arraydf)
    loaded_model = pickle.load(open("model_lr.pkl", "rb")) 
    result = loaded_model.predict(to_predict) 
    return result[0] 

@app.route('/prediction', methods = ['POST']) 
def result(): 
    if request.method == 'POST': 
        to_predict_list = request.form.to_dict() 
        to_predict_list = list(to_predict_list.values()) 
        to_predict_list = list(map(int, to_predict_list)) 
        result = ValuePredictor(to_predict_list)         
        if int(result)== 1:
            template = render_template("resultsurvived.html")
        elif int(result) == 0: 
            template = render_template("resultwasted.html")
        return template 
