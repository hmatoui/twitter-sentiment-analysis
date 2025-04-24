import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, LSTM, Dense


import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("../logs/training.log"),  # Log to a file
        logging.StreamHandler()                        # Log to stdout
    ]
)


data = pd.read_csv('../data/processed/tweet_embeddings.csv')
# Convert embeddings from JSON strings back to numpy arrays
data['embedding'] = data['embedding'].apply(lambda x: np.array(json.loads(x)))
X = np.array(data['embedding'].tolist())  # Assuming embeddings are stored as lists
y = np.array((data['target']==4).astype(int).tolist())  # Binary target labels (0 or 1)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


from gensim.models import Word2Vec

# Load Word2Vec model
word2vec_model = Word2Vec.load('../models/word2vec/word2vec.model')

# Get the Word2Vec embedding dimension
embedding_dim = word2vec_model.vector_size
print("Embedding Dimension", embedding_dim)
# Create an embedding matrix
vocab_size = len(word2vec_model.wv.key_to_index) + 1
print("Total words", vocab_size)
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, index in word2vec_model.wv.key_to_index.items():
    embedding_matrix[index] = word2vec_model.wv[word]

# Define the LSTM model
model = Sequential()

# Embedding layer
embedding_layer = Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    input_length=X_train.shape[1],  # Number of tokens per tweet
    trainable=False  # Freeze the embeddings
)
model.add(embedding_layer)

# Add LSTM and Dense layers
model.add(Dropout(0.5))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.build(input_shape=(None, X_train.shape[1]))  # Batch size is flexible

# Display model summary
model.summary()


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
logging.info("Training started...")
# Check if TensorFlow is using GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Available devices:")
print(device_lib.list_local_devices())

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64,
    verbose=1,
)
logging.info("Training completed successfully.")


# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Generate classification report
from sklearn.metrics import classification_report

y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))


model.save('../models/sentiment/sentiment_model.h5')
model.save('../app/model/sentiment_model.h5')