from flask import Flask,render_template,request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import pickle


app = Flask(__name__, template_folder='tempelates')


ridgeModel = pickle.load(open('pickleFiles/ridge.pkl','rb'))
scalerModel = pickle.load(open('pickleFiles/scaler.pkl','rb'))

@app.route("/")
def welcome():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = int(request.form.get('Classes'))
        Region = int(request.form.get('Region'))

        data = scalerModel.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridgeModel.predict(data)

        return render_template("prediction.html", results=result[0])  
    else:
        return render_template("prediction.html")  

       

if __name__ == "__main__":
    app.run(debug=True,port=4040)