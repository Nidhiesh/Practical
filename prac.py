import numpy as np 
import random
import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report 
import matplotlib.pyplot as plt 

# 1. Generate Synthetic Email Data (100% Offline, no downloads)
ham_words = ["meeting", "project", "hello", "tomorrow", "submit", "good", "morning", "team", "review", "report"]
spam_words = ["free", "win", "cash", "prize", "click", "claim", "urgent", "money", "guaranteed", "offer"]

texts = []
labels = []

# Generate 500 fake Ham (0) and Spam (1) emails
for _ in range(500):
    texts.append(" ".join(random.sample(ham_words, k=5)))
    labels.append(0)
    
    texts.append(" ".join(random.sample(spam_words, k=5)))
    labels.append(1)

# 2. Preprocess Text (Tokenization and Padding)
MAX_VOCAB = 1000 
MAX_LEN = 10 

tokenizer = Tokenizer(num_words=MAX_VOCAB) 
tokenizer.fit_on_texts(texts) 
sequences = tokenizer.texts_to_sequences(texts) 
X = pad_sequences(sequences, maxlen=MAX_LEN) 
y = np.array(labels) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# 3. Build the GRU Model for Text Classification
model = Sequential([ 
    Embedding(input_dim=MAX_VOCAB, output_dim=32, input_length=MAX_LEN), 
    GRU(32, return_sequences=False), 
    Dropout(0.3), 
    Dense(16, activation="relu"), 
    Dense(1, activation="sigmoid") # Sigmoid for Binary Classification (Spam vs Ham)
]) 

# 4. Compile the Model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]) 

# 5. Train the Model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, verbose=1) 

# 6. Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0) 
print(f"\nTest Accuracy: {acc:.4f}\n") 

preds = (model.predict(X_test) > 0.5).astype("int32") 
print(classification_report(y_test, preds, target_names=["Ham", "Spam"])) 

# 7. Visualize (Normal & Simple Plotting)
plt.plot(history.history["accuracy"], label="Train Acc") 
plt.plot(history.history["val_accuracy"], label="Val Acc") 
plt.title("Model Accuracy") 
plt.legend() 
plt.show() 

plt.plot(history.history["loss"], label="Train Loss") 
plt.plot(history.history["val_loss"], label="Val Loss") 
plt.title("Model Loss") 
plt.legend() 
plt.show()