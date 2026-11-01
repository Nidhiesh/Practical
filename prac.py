import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches

# ==========================================
# STEP 1 & 2: GENERATE & PREPROCESS DATA (OFFLINE)
# ==========================================
num_samples = 500
img_size = 64
X = np.zeros((num_samples, img_size, img_size, 1), dtype=np.float32)
y_boxes = np.zeros((num_samples, 4), dtype=np.float32) # Target: [x, y, width, height]

for i in range(num_samples):
    # Create a random "car" (a white box on a dark background)
    w, h = np.random.randint(10, 20, 2)
    x, y = np.random.randint(10, img_size - 20, 2)
    
    # Draw the car onto the image matrix
    X[i, y:y+h, x:x+w, 0] = 1.0 
    
    # Save the normalized box coordinates (between 0 and 1) for the model to learn
    y_boxes[i] = [x / img_size, y / img_size, w / img_size, h / img_size]

# Split into training and testing sets
split = 400
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_boxes[:split], y_boxes[split:]

# ==========================================
# STEP 3: BUILD THE DNN (CNN)
# ==========================================
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(img_size, img_size, 1)),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    
    # Output Layer: 4 neurons to predict Bounding Box coordinates [x, y, w, h]
    tf.keras.layers.Dense(4, activation='sigmoid') 
])

# ==========================================
# STEP 4 & 5: COMPILE AND TRAIN
# ==========================================
model.compile(optimizer='adam', loss='mse')
print("Training Simple Object Detection Model...")
model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=1)

# ==========================================
# STEP 6 & 7: TEST AND DISPLAY BOUNDING BOX
# ==========================================
test_img = X_test[0]
pred_box = model.predict(test_img.reshape(1, img_size, img_size, 1))[0]

# Convert the predicted coordinates (0 to 1) back to actual pixel sizes (0 to 64)
x, y, w, h = pred_box * img_size

# Draw the image and the bounding box using Matplotlib
fig, ax = plt.subplots(1)
ax.imshow(test_img[:, :, 0], cmap='gray')

# Create a red rectangle patch 
rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
ax.add_patch(rect)

plt.title("Traffic Object Detected")
plt.axis('off')
plt.show()