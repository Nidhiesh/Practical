import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 

print("Generating synthetic offline leaf images...")

# 1. Generate Synthetic Leaf Data (Offline)
# 0 = Healthy Leaf, 1 = Diseased Leaf
num_samples = 400
img_size = 64
X = np.zeros((num_samples, img_size, img_size, 3), dtype=np.float32)
y = np.random.randint(0, 2, num_samples)
class_names = ["Healthy", "Diseased"]

for i in range(num_samples):
    # Create a base green "leaf" (higher values in the green channel)
    X[i, :, :, 1] = np.random.uniform(0.5, 0.8) # Green
    X[i, :, :, 0] = np.random.uniform(0.1, 0.3) # Red
    X[i, :, :, 2] = np.random.uniform(0.1, 0.3) # Blue
    
    if y[i] == 1: 
        # If diseased, simulate infection by drawing random brown/black spots
        num_spots = np.random.randint(3, 8)
        for _ in range(num_spots):
            sx, sy = np.random.randint(5, img_size-5, 2)
            X[i, sx:sx+4, sy:sy+4, 0] = 0.6  # Brownish-red
            X[i, sx:sx+4, sy:sy+4, 1] = 0.3
            X[i, sx:sx+4, sy:sy+4, 2] = 0.1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build a Custom CNN from Scratch
# This completely replaces the heavy MobileNetV2 architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(img_size, img_size, 3)),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2, activation='softmax') # 2 classes: Healthy vs Diseased
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 3. Train the CNN
print("Training Custom CNN for Leaf Disease Detection...")
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# 4. Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc:.4f}\n")

# 5. Visualize Predictions on 5 sample images
preds = model.predict(X_test[:5])

plt.figure(figsize=(12, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[i])
    
    pred_class = class_names[np.argmax(preds[i])]
    true_class = class_names[y_test[i]]
    color = "green" if pred_class == true_class else "red"
    
    plt.title(f"Pred: {pred_class}\nTrue: {true_class}", color=color, fontsize=10)
    plt.axis("off")

plt.tight_layout()
plt.show()