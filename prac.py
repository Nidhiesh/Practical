import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder 

print("Generating synthetic offline MFCC features...")

# 1. Generate Synthetic Audio Features (Bypasses GitHub download and librosa)
np.random.seed(42)
num_samples = 1000
num_features = 40  # Represents 40 MFCC features from an audio file
num_classes = 10   # Representing spoken digits 0 through 9

X = np.zeros((num_samples, num_features))
y = np.random.randint(0, num_classes, num_samples)

# Adding a mathematical pattern based on the label so the ANN can actually learn
for i in range(num_samples):
    X[i] = np.random.normal(loc=y[i]*0.2, scale=1.0, size=num_features)

X = np.array(X) 
y = np.array(y) 
print(f"Feature extraction complete! Shape: {X.shape}") 

# 2. Preprocess the Data
le = LabelEncoder() 
y_encoded = le.fit_transform(y) 

X_train, X_test, y_train, y_test = tts(
    X, y_encoded, test_size=0.2, random_state=42 
) 
print(f"Data split: Train = {len(X_train)} , Test = {len(X_test)}") 

# 3. Build the ANN Model
model = tf.keras.models.Sequential([ 
    tf.keras.layers.Dense(256, activation='relu', input_shape=(40,)), 
    tf.keras.layers.Dropout(0.3), 
    tf.keras.layers.Dense(128, activation='relu'), 
    tf.keras.layers.Dropout(0.3), 
    tf.keras.layers.Dense(10, activation='softmax') 
]) 

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
model.summary() 

# 4. Train the Model
print("Training the model...") 
history = model.fit(X_train, y_train, epochs=40, batch_size=16, validation_data=(X_test, y_test), verbose=1) 

# 5. Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test) 
print(f"\nTest Accuracy: {test_acc * 100:.2f}%\n") 

# 6. Plotting
plt.figure(figsize=(10,4)) 

plt.subplot(1,2,1) 
plt.plot(history.history['accuracy'], label='Train Acc') 
plt.plot(history.history['val_accuracy'], label='Val Acc') 
plt.legend() 
plt.title('Model Accuracy') 

plt.subplot(1,2,2) 
plt.plot(history.history['loss'], label='Train Loss') 
plt.plot(history.history['val_loss'], label='Val Loss') 
plt.legend() 
plt.title('Model Loss') 

plt.tight_layout()
plt.show() 

# 7. Predict on a simulated new audio file
def predict_audio(synthetic_feature): 
    feature = np.expand_dims(synthetic_feature, axis=0) 
    pred = model.predict(feature) 
    predicted_label = le.inverse_transform([np.argmax(pred)]) 
    print(f"🎤 Predicted Digit: {predicted_label[0]}") 

print("Simulating prediction on a new audio file...")
# Grab one sample from our test set to act as the "new" audio file
sample_feature = X_test[0]
predict_audio(sample_feature)