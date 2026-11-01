import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ==========================================
# STEP 1: GENERATE SYNTHETIC CARTOON DATA (OFFLINE)
# ==========================================
# Creating 1000 simple 28x28 grayscale "faces" so it runs without internet
num_samples = 1000
img_size = 28
X_train = np.zeros((num_samples, img_size, img_size, 1), dtype=np.float32)

for i in range(num_samples):
    # Draw a simple face pattern (left eye, right eye, and mouth)
    X_train[i, 8:12, 8:12, 0] = 1.0   # Left eye
    X_train[i, 8:12, 16:20, 0] = 1.0  # Right eye
    X_train[i, 20:22, 10:18, 0] = 1.0 # Mouth
    # Add slight random noise so the GAN has to learn variations
    X_train[i] += np.random.uniform(0, 0.1, (img_size, img_size, 1))

X_train = np.clip(X_train, 0, 1)

# ==========================================
# STEP 2: BUILD GENERATOR & DISCRIMINATOR
# ==========================================
latent_dim = 50

# Generator: Takes random noise and creates an image
generator = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(latent_dim,)),
    layers.Dense(img_size * img_size, activation='sigmoid'),
    layers.Reshape((img_size, img_size, 1))
])

# Discriminator: Takes an image and guesses Real (1) or Fake (0)
discriminator = models.Sequential([
    layers.Flatten(input_shape=(img_size, img_size, 1)),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid') 
])

# ==========================================
# STEP 3: COMPILE MODELS
# ==========================================
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# Create combined GAN model (Generator + Discriminator)
discriminator.trainable = False # Freeze discriminator while training the generator
gan = models.Sequential([generator, discriminator])
gan.compile(optimizer='adam', loss='binary_crossentropy')

# ==========================================
# STEP 4 & 6: TRAIN BOTH NETWORKS ALTERNATELY
# ==========================================
epochs = 50
batch_size = 32

print("Training GAN... This will take a few seconds.")
for epoch in range(epochs):
    
    # --- 1. Train Discriminator ---
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_imgs = generator.predict(noise, verbose=0)
    
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]
    
    # Train discriminator on real images (target=1) and fake images (target=0)
    d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))
    
    # --- 2. Train Generator ---
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    
    # We want the generator to fool the discriminator, so we tell it the target is 1 (Real)
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1))) 

# ==========================================
# STEP 5 & 7: EVALUATE & DISPLAY GENERATED IMAGES
# ==========================================
print("Generating new cartoon faces...")
test_noise = np.random.normal(0, 1, (5, latent_dim))
generated_images = generator.predict(test_noise, verbose=0)

# Simple Matplotlib plotting
fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    axes[i].imshow(generated_images[i, :, :, 0], cmap='gray')
    axes[i].axis('off')
    
plt.suptitle("Generated Cartoon Faces")
plt.show()