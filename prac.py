import numpy as np 
import pandas as pd 
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.datasets import make_classification 
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 

# 1. Generate Synthetic Share Market Data (100% Offline)
# Simulating 'price' and 'volume' where some transactions are normal (0) and some are fraudulent (1)
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=42) 

# 2. Preprocess Data
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 

# LSTMs require 3D input: (samples, time_steps, features). Here time_steps=1, features=2
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, 2)) 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y) 

# 3. Build the LSTM Model for Fraud Detection
model = Sequential([ 
    Input(shape=(1, 2)), 
    LSTM(32, return_sequences=False), 
    Dropout(0.3), 
    Dense(16, activation="relu"), 
    Dense(1, activation="sigmoid") # Sigmoid outputs probability of Fraud (0 to 1)
]) 

# 4. Compile the Model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]) 

# 5. Train the Model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=12, batch_size=32, verbose=1) 

# 6. Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0) 
print(f"\nTest accuracy: {acc:.4f}\n") 

y_pred = (model.predict(X_test) > 0.5).astype("int32") 
print(classification_report(y_test, y_pred)) 
print("ROC AUC:", roc_auc_score(y_test, y_pred))

# 7. Visualize (Normal & Simple Plotting)

# Show Accuracy Graph
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Model Accuracy")
plt.legend()
plt.show()

# Show Loss Graph
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.legend()
plt.show()

# Show Confusion Matrix (No Seaborn required)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
plt.title("Fraud Detection Confusion Matrix")
plt.show()