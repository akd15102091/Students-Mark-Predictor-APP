import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template
app = Flask(__name__)
model = joblib.load('student_mark_predictor.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = np.array(input_features)

    if(input_features[0] <0  or input_features[0]>24) :
        return render_template('index.html', prediction_text='Please Enter the valid hours(between 0-24)')

    output = model.predict([features_value])[0][0].round(2)

    if(output>100) :
        output = 99

    return render_template('index.html', prediction_text='You will get [{}%] marks, when you do study [{}] hours per day '.format(output,input_features[0]))

if __name__ == "__main__":
    app.run(debug=True)
