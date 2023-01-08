import glob
import joblib
import numpy as np
import os
import pickle
from PIL import Image
import re
import sys
import tensorflow as tf 

from flask import Flask, redirect, url_for, request, render_template,flash,redirect
from flask import request
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from werkzeug.utils import secure_filename


app = Flask(__name__)
MODEL_PATH = 'models/BrainTumour.h5'
model = load_model(MODEL_PATH)



MODEL_PATH2 = 'models/Pneumonia.h5'
model2 = load_model(MODEL_PATH2)
print('Model loaded. Start serving...')


filename = 'models/heart.pkl'
model3 = pickle.load(open(filename, 'rb'))


def predict_label(img_path, model):
    img = image.load_img(img_path, target_size=(200,200)) 
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')/255
    preds = model.predict(img)
    pred = np.argmax(preds,axis = 1)
    str0 = ''
    if pred[0] == 0:
         str0 = "Glioma"
    elif pred[0] == 1:
        str0 = 'Meningioma'
    elif pred[0]==3:
        str0 = 'Pituitary'
    else:
        str0 = "Normal"
    return str0

def model_predict(img_path):
	img = tf.keras.utils.load_img(img_path, target_size=(64, 64))
	img = tf.keras.preprocessing.image.img_to_array(img)
	img = np.expand_dims(img, axis=0)
	preds = model2.predict(img)
	if preds==1:
		preds ="Pneumonia"
	else:
		preds="Normal"
	return preds


@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("main.html")

@app.route("/index", methods=['GET', 'POST'])
def index():
	return render_template("index.html")

@app.route("/index2", methods=['GET', 'POST'])
def index2():
	return render_template("index2.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static/brain_uploads" + img.filename	
		img.save(img_path)
		p = predict_label(img_path,model)
	return render_template("index.html", prediction = p, img_path = img_path)


@app.route("/predict", methods = ['GET', 'POST'])
def output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static/pneumonia_uploads" + img.filename	
		img.save(img_path)
		p = model_predict(img_path)
	return render_template("index2.html", prediction = p, img_path = img_path)


@app.route("/liver")
def cancer():
    return render_template("liver.html")

def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==7):
        loaded_model = joblib.load('models/liver.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/output', methods = ["POST"])
def predict():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
         #liver
        if(len(to_predict_list)==7):
            result = ValuePredictor(to_predict_list,7)
    
    if(int(result)==1):
        prediction = "Sorry you have chances of getting the disease. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
    return(render_template("result.html", prediction_text=prediction)) 

@app.route("/cardio")
def cardio():
    return render_template("cardio.html")

@app.route('/p', methods=['GET','POST'])
def p():
    if request.method == 'POST':

        age = int(request.form['age'])
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = request.form.get('exang')
        oldpeak = float(request.form['oldpeak'])
        slope = request.form.get('slope')
        ca = int(request.form['ca'])
        thal = request.form.get('thal')
        
        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        my_prediction = model3.predict(data)
        
        return render_template('res.html', prediction=my_prediction)


if __name__ == '__main__':
        app.run(host="0.0.0.0",port=8000)
    
