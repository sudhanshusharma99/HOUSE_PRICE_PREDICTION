import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)
model = pickle.load(open('hr_linearregression.pkl','rb')) 

port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port, debug=True)

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    exp = float(request.args.get('exp'))
    
    prediction = model.predict([[exp]])
    
        
    return render_template('index.html', prediction_text='Regression Model  has predicted House Price for given SqFt is : {}'.format(prediction))


app.run(debug=True)