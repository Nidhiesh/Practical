import numpy as np 
import matplotlib.pyplot as plt 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split 
 
# 1. Generate Synthetic Data (100% Offline)
np.random.seed(42) 
time_steps = 500 
t = np.arange(time_steps) 
stock_prices = 50 + 0.1 * t + 10 * np.sin(0.1 * t) + np.random.normal(0, 2, time_steps) 
 
# Visualize the original data
plt.plot(t, stock_prices) 
plt.title("Synthetic Stock Prices") 
plt.xlabel("Time") 
plt.ylabel("Price") 
plt.show() 
 
# 2. Preprocess the Data
scaler = MinMaxScaler(feature_range=(0, 1)) 
stock_prices_scaled = scaler.fit_transform(stock_prices.reshape(-1, 1)) 
 
def create_dataset(data, look_back=10): 
    X, y = [], [] 
    for i in range(len(data) - look_back): 
        X.append(data[i:i+look_back]) 
        y.append(data[i+look_back]) 
    return np.array(X), np.array(y) 
 
look_back = 10 
X, y = create_dataset(stock_prices_scaled, look_back) 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) 
 
print("X_train shape:", X_train.shape) 
print("y_train shape:", y_train.shape) 
 
# 3. Build the LSTM Model
model = Sequential([ 
    LSTM(128, return_sequences=True, input_shape=(look_back, 1)), 
    LSTM(64), 
    Dense(1) 
]) 
 
model.compile(optimizer='adam', loss='mse') 

# 4. Train the Model
history = model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1) 

# Visualize Training Loss
plt.plot(history.history['loss'], label='Training Loss') 
plt.title("Overfitting Demonstration: Training Loss") 
plt.xlabel("Epoch") 
plt.ylabel("Loss") 
plt.legend() 
plt.show() 

# 5. Evaluate and Predict
test_loss = model.evaluate(X_test, y_test) 
print(f"Test loss: {test_loss}") 

y_pred = model.predict(X_test) 
y_test_inv = scaler.inverse_transform(y_test) 
y_pred_inv = scaler.inverse_transform(y_pred) 

# Visualize Predictions vs True Data
plt.plot(y_test_inv, label='True Stock Price') 
plt.plot(y_pred_inv, label='Predicted Stock Price') 
plt.title("Stock Price Prediction on Test Set") 
plt.xlabel("Time Step") 
plt.ylabel("Price") 
plt.legend() 
plt.show()