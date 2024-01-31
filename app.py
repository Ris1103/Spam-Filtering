from flask import Flask, render_template, request, jsonify
import pickle

tokenizer = pickle.load(open('models/cv.pkl', 'rb'))
model = pickle.load(open('models/clf.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

#This is where we will receive the data from the form and send it to the model
@app.route('/predict', methods = ['POST'])
def predict():
    email = request.form.get('content')
    # Check if input is a string
    if isinstance(email, str):
        tokenized_email = tokenizer.transform([email])
    else:
        tokenized_email = tokenizer.transform(email)
    prediction = model.predict(tokenized_email)
    prediction = 1 if prediction == 1 else -1
    return render_template('index.html', prediction = prediction, email = email)

if __name__ == '__main__':
    app.run(debug=True)