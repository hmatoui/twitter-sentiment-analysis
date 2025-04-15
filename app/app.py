from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and tokenizer
model = tf.keras.models.load_model('../models/sentiment_model/sentiment_model.h5')
with open('../models/tokenizer/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Define sentiment mapping
sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

@app.route('/')
def home():
    return render_template('index.html')  # HTML template for input

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json if request.is_json else request.form
        text = data.get('text', '')

        # Tokenize input
        seq = tokenizer.texts_to_sequences([text])
        padded_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)

        # Predict sentiment
        probabilities = model.predict(padded_seq)[0]
        sentiment_idx = np.argmax(probabilities)
        sentiment = sentiment_map[sentiment_idx]

        return jsonify({
            'text': text,
            'sentiment': sentiment,
            'probabilities': {sentiment_map[i]: float(prob) for i, prob in enumerate(probabilities)}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
