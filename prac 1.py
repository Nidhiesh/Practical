import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# 1. Load Data (Downloads an 11MB file once, then works 100% offline)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Preprocess Data
# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape from 2D images (28, 28) to 1D arrays (784,) for the ANN
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# 3. Create ANN Model
model = models.Sequential([
    layers.Input(shape=(28*28,)),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax') # 10 classes for digits 0-9
])

# 4. Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 5. Train Model
# Using validation_split allows you to see accuracy on unseen data during training
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# 6. Evaluate Model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc*100:.2f}%\n")

# 7. Test and Display Predictions
preds = model.predict(x_test[:10])
preds_cls = np.argmax(preds, axis=1)

plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    # Reshape the 1D array back to 28x28 for plotting
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"True: {y_test[i]} | Pred: {preds_cls[i]}")
    plt.axis('off')
    
plt.tight_layout()
plt.show()