import numpy as np 
import random
import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 

# 1. Generate Synthetic Sentiment Data (100% Offline)
pos_words = ["good", "great", "awesome", "fantastic", "love", "amazing", "excellent"]
neg_words = ["bad", "terrible", "awful", "hate", "boring", "worst", "horrible"]

texts = []
labels = []

# Generate 500 fake Positive (1) and Negative (0) reviews
for _ in range(500):
    texts.append(" ".join(random.sample(pos_words, k=4)))
    labels.append(1) 
    
    texts.append(" ".join(random.sample(neg_words, k=4)))
    labels.append(0) 

# 2. Preprocess Data
vocab_size = 50 
maxlen = 10 

tokenizer = Tokenizer(num_words=vocab_size) 
tokenizer.fit_on_texts(texts) 
sequences = tokenizer.texts_to_sequences(texts) 
X = pad_sequences(sequences, maxlen=maxlen) 
y = np.array(labels) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# 3. Build the LSTM Model
model = Sequential([ 
    Embedding(input_dim=vocab_size, output_dim=32, input_length=maxlen), 
    LSTM(32, return_sequences=False), 
    Dropout(0.3), 
    Dense(1, activation='sigmoid') # Binary Classification (Positive or Negative)
]) 

# 4. Compile the Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

# 5. Train the Model
print("Training LSTM Sentiment Model...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, verbose=1) 

# 6. Evaluate and Predict
loss, accuracy = model.evaluate(X_test, y_test, verbose=0) 
print(f"\nTest Accuracy: {accuracy*100:.2f}%\n") 

# Test a custom sentence
sample_review = "fantastic love amazing"
sample_seq = tokenizer.texts_to_sequences([sample_review])
sample_input = pad_sequences(sample_seq, maxlen=maxlen)

prediction = model.predict(sample_input)[0][0]
sentiment = "Positive 🎤" if prediction > 0.5 else "Negative 🎤"
print(f"Review: '{sample_review}'")
print(f"Predicted Sentiment: {sentiment}\n")

# 7. Visualize (Normal & Simple Plotting)
plt.plot(history.history['accuracy'], label='Train Acc') 
plt.plot(history.history['val_accuracy'], label='Val Acc') 
plt.title('Model Accuracy') 
plt.legend() 
plt.show() 

plt.plot(history.history['loss'], label='Train Loss') 
plt.plot(history.history['val_loss'], label='Val Loss') 
plt.title('Model Loss') 
plt.legend() 
plt.show()