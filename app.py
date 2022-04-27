# Importing essential libraries
from flask import Flask, render_template, request, jsonify, redirect
from flask_cors import CORS
import pickle
# Load the Naive Bayes model and TfidfVectorizer object from disk
filename = 'Movies_Review_Classification.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('count-Vectorizer.pkl','rb'))
app = Flask(__name__, template_folder='src')
CORS(app)

@app.route('/')
def home():
	return render_template('App.js')

@app.route('/predict',methods=['POST','GET'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		global my_prediction
		my_prediction = classifier.predict(vect)
		url = "http://127.0.0.1:3000/s"
		return redirect(url,code= 302)
	
	if request.method == 'GET':
		if my_prediction ==1:
			prediction= "Positive Review"
		elif my_prediction==0:
			prediction= "Negative Review"
		#prediction  =  prediction 
		return {'prediction': prediction}

if __name__ == '__main__':
	app.run(debug=True)