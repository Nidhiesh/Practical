import pandas as pd 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.metrics import classification_report 

# 1. Generate Simplified Synthetic Offline Data (3 columns total)
np.random.seed(42)
n_samples = 1000

data = {
    "Temperature (°C)": np.random.normal(37.0, 0.5, n_samples),
    "Heart_Rate (bpm)": np.random.normal(75, 12, n_samples),
    "Target_Health_Status": np.random.choice(['Normal', 'Critical'], n_samples)
}
df = pd.DataFrame(data)

# 2. Preprocess the Data
feature_cols = [ 
    "Temperature (°C)", "Heart_Rate (bpm)" 
] 
target_col = "Target_Health_Status" 

X = df[feature_cols].values 
y = df[target_col].values 

label_encoder = LabelEncoder() 
y_encoded = label_encoder.fit_transform(y) 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 

X_train, X_test, y_train_raw, y_test_raw = train_test_split( 
    X_scaled, y_encoded, test_size=0.2, random_state=42 
) 

# Convert labels to categorical (one-hot encoding) for Softmax
y_train_cat = tf.keras.utils.to_categorical(y_train_raw) 
y_test_cat = tf.keras.utils.to_categorical(y_test_raw) 

# 3. Build the ANN Model
model = tf.keras.models.Sequential([ 
    tf.keras.layers.Input(shape=(X_train.shape[1],)), 
    tf.keras.layers.Dense(16, activation='relu'), 
    tf.keras.layers.Dense(8, activation='relu'), 
    tf.keras.layers.Dense(2, activation='softmax')  # 2 output classes
]) 

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 

# 4. Train the Model
history = model.fit(X_train, y_train_cat, epochs=46, batch_size=16, validation_split=0.2, verbose=1) 

# 5. Evaluate the Model
loss, acc = model.evaluate(X_test, y_test_cat) 
print(f"\nTest Accuracy: {acc:.4f}") 

# 6. Predictions and Classification Report
y_pred = model.predict(X_test) 
y_pred_classes = np.argmax(y_pred, axis=1) 

print("\nClassification Report:\n") 
print(classification_report(y_test_raw, y_pred_classes, target_names=label_encoder.classes_)) 

# 7. Plotting Accuracy and Loss
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy') 
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') 
plt.xlabel('Epoch') 
plt.ylabel('Accuracy') 
plt.title('Model Accuracy') 
plt.legend() 
plt.grid(True) 

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss') 
plt.plot(history.history['val_loss'], label='Validation Loss') 
plt.xlabel('Epoch') 
plt.ylabel('Loss') 
plt.title('Model Loss') 
plt.legend() 
plt.grid(True) 

plt.tight_layout()
plt.show()