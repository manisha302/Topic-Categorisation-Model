# import numpy as np
from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import Preprocess
import gunicorn


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('Transform.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():

	if request.method == 'POST':
		message = request.form['message']
		data = Preprocess.preprocess(text1=	message)
		print(data)
		data = [data]
		# print(data)
		vect = cv.fit_transform(data)
		# print(vect)
		my_prediction = model.fit_transform(vect)
		sentiment = int(np.argmax(my_prediction))
	if sentiment == 0:
		Prediction = 'Mobile Recharge & Bill Payment'
	elif sentiment == 1:
		Prediction = 'Customer Services'
	elif sentiment == 2:
		Prediction = 'Payment Related Issues'
	elif sentiment == 3:
		Prediction = 'Offers & Cashback'
	elif sentiment == 4:
		Prediction = 'App Upgradation'
	elif sentiment == 5:
		Prediction = 'User Interface'
	elif sentiment == 6:
		Prediction = 'Transaction Limit'
	else:
		Prediction = 'Invalid Input'
	return render_template('result1.html',prediction_text = Prediction)

if __name__ == "__main__":
    app.run(debug=True)
