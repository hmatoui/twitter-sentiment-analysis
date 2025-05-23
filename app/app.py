from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import pickle
import torch

app = Flask(__name__)

# Load the trained model and tokenizer
model = tf.keras.models.load_model('../models/sentiment/sentiment_model.h5')
with open('../models/tokenizer/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict(tweet):
    try:
        # Tokenize input
        inputs = tokenizer(tweet, return_tensors="pt", truncation=True, max_length=512)

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Map predictions to labels
        predicted_label = torch.argmax(probabilities, dim=-1).item()
        labels = ["Negative", "Neutral", "Positive"]

        return labels[predicted_label]
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = ""
    tweet = ""
    if request.method == 'POST':
        tweet = request.form['tweet']
        prediction = predict(tweet) if tweet!="" else ""
    return render_template('index.html', tweet=tweet, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
