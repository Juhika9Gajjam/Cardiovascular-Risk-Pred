import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model_cls = pickle.load(open('cardio_model_GB.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    #For rendering results on HTML GUI
    li=[]
    for i in request.form.values():
        i=str(i)
        if i.isalpha()==True:
            if i.lower()=='m' or i.lower()=='yes':
                li.append(1)
            else:
                li.append(0)
        else:
            li.append(float(i))
    li=[li]
    features=pd.DataFrame(data=li)
    #final_features=ss.transform(final_features)
    prediction = model_cls.predict(features)

    #output = round(prediction[0], 2)
    #flag= prediction[0]
    flag= prediction[0]
    if flag:
        #return render_template('index.html', prediction_text=flag)
        return render_template('index.html', prediction_text='Fantastic! You have no cardiovascular risk ahead.')
    else:
        #return render_template('index.html', prediction_text=flag)
        return render_template('index.html', prediction_text='There is a possibility of cardiovascular risk in your future.')

if __name__ == "__main__":
    app.run(debug=True)