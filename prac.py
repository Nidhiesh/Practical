import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split

# 1. Generate Synthetic Audio Data (100% Offline)
# Simulating 1000 audio frames, each containing 100 sequential features
num_samples = 1000
feature_dim = 100

# Generate Clean "Audio" (Smooth sine waves representing clear voice frequencies)
x_axis = np.linspace(0, 10, feature_dim)
clean_audio = np.array([np.sin(x_axis + i) for i in range(num_samples)])

# Generate Noisy "Audio" (Adding random static to the clean audio)
noise_factor = 0.5
noisy_audio = clean_audio + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=clean_audio.shape)
noisy_audio = np.clip(noisy_audio, -1.0, 1.0) 

# Split into training and testing sets
X_train_noisy, X_test_noisy, y_train_clean, y_test_clean = train_test_split(
    noisy_audio, clean_audio, test_size=0.2, random_state=42
)

# 2 & 3. Build the Autoencoder (Encoder + Decoder)
model = Sequential([
    Input(shape=(feature_dim,)),
    
    # Encoder: Compresses the audio signal to find the core patterns
    Dense(64, activation='relu'),
    Dense(32, activation='relu'), 
    
    # Decoder: Reconstructs the audio signal while leaving the random noise behind
    Dense(64, activation='relu'),
    Dense(feature_dim, activation='linear') 
])

# 4. Compile the Model (Using Mean Squared Error for regression/reconstruction)
model.compile(optimizer='adam', loss='mse')

# 5. Train the Model (Teach it to map Noisy input -> Clean target)
print("Training Audio Autoencoder...")
history = model.fit(
    X_train_noisy, y_train_clean, 
    validation_data=(X_test_noisy, y_test_clean), 
    epochs=20, 
    batch_size=32, 
    verbose=1
)

# 6. Evaluate
test_loss = model.evaluate(X_test_noisy, y_test_clean, verbose=0)
print(f"\nTest MSE Loss: {test_loss:.4f}\n")

# 7. Reconstruct/Output the denoised audio
denoised_audio = model.predict(X_test_noisy)

# ==========================================
# VISUALIZATION (Simple Sequential Plotting)
# ==========================================

# Plot Loss Curve
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Autoencoder Training Loss')
plt.legend()
plt.show()

# Plot 1: The original clean signal
plt.plot(y_test_clean[0], label='Clean Signal', color='green')
plt.title("Original Clean Audio")
plt.legend()
plt.show()

# Plot 2: The signal corrupted by static noise
plt.plot(X_test_noisy[0], label='Noisy Input', color='red')
plt.title("Audio with Background Noise")
plt.legend()
plt.show()

# Plot 3: The model's attempt to clean the noise
plt.plot(denoised_audio[0], label='Denoised Output', color='blue')
plt.title("Autoencoder Denoised Output")
plt.legend()
plt.show()