import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# ==================== DATA LOADING AND PREPROCESSING ====================

def load_and_preprocess_data(file_path, symbol='ASIANPAINT'):
    """
    Load and preprocess stock market data, focusing on 'Close' column
    """
    print(f"Loading data for {symbol}...")
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading file for {symbol}: {e}")
        return None

    # Standardize column names
    column_mapping = {
        'Date': 'Date', 'date': 'Date', 'DATE': 'Date',
        'Close': 'Close', 'close': 'Close', 'CLOSE': 'Close'
    }

    for old_name, new_name in column_mapping.items():
        if old_name in data.columns:
            data.rename(columns={old_name: new_name}, inplace=True)

    required_cols = ['Date', 'Close']
    if not all(col in data.columns for col in required_cols):
        print(f"Missing required columns for {symbol}. Available columns:", data.columns.tolist())
        return None

    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')
    data.reset_index(drop=True, inplace=True)
    data = data.fillna(method='ffill').fillna(method='bfill')

    print(f"Data loaded successfully for {symbol}. Shape: {data.shape}")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    return data

def visualize_data(data, symbol):
    """
    Visualize the Close price over time with moving averages
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')

    ma_day = [10, 50, 100]
    for ma in ma_day:
        column_name = f'MA for {ma} days'
        data[column_name] = data['Close'].rolling(window=ma).mean()
        plt.plot(data['Date'], data[column_name], label=column_name, alpha=0.7)

    plt.title(f'Close Price with Moving Averages for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ==================== DATA PREPARATION ====================

def create_dataset(dataset, time_step=100):
    """
    Convert an array of values into a dataset matrix for univariate prediction
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# ==================== MODEL ARCHITECTURE ====================

def create_advanced_model(input_shape):
    """
    Create a stacked LSTM model for stock price prediction
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
    return model

# ==================== MAIN EXECUTION ====================

def train_and_evaluate_model(data, symbol, time_step=100):
    """
    Train and evaluate model for a specific stock, saving model and scaler
    """
    # Extract and scale 'Close' column
    close_data = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    close_scaled = scaler.fit_transform(close_data)

    # Prepare sequences
    X, y = create_dataset(close_scaled, time_step)

    # Split data into train and test sets (80% train, 20% test)
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Reshape for LSTM input [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    print(f"Training data for {symbol}: {X_train.shape}, {y_train.shape}")
    print(f"Testing data for {symbol}: {X_test.shape}, {y_test.shape}")

    # Create and train the model
    model = create_advanced_model((time_step, 1))
    print(f"Model summary for {symbol}:")
    print(model.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=5,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training and Validation Loss for {symbol}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Rescale predictions and actual values
    train_predictions_rescaled = scaler.inverse_transform(train_predictions)
    test_predictions_rescaled = scaler.inverse_transform(test_predictions)
    y_train_rescaled = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics for training set
    print(f"Training Metrics for {symbol}:")
    mse_train = mean_squared_error(y_train_rescaled, train_predictions_rescaled)
    mae_train = mean_absolute_error(y_train_rescaled, train_predictions_rescaled)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train_rescaled, train_predictions_rescaled)
    mape_train = np.mean(np.abs((y_train_rescaled - train_predictions_rescaled) / y_train_rescaled)) * 100
    print(f"MSE: {mse_train:.4f}")
    print(f"MAE: {mae_train:.4f}")
    print(f"RMSE: {rmse_train:.4f}")
    print(f"R² Score: {r2_train:.4f}")
    print(f"MAPE: {mape_train:.2f}%")

    # Calculate metrics for test set
    print(f"Testing Metrics for {symbol}:")
    mse_test = mean_squared_error(y_test_rescaled, test_predictions_rescaled)
    mae_test = mean_absolute_error(y_test_rescaled, test_predictions_rescaled)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test_rescaled, test_predictions_rescaled)
    mape_test = np.mean(np.abs((y_test_rescaled - test_predictions_rescaled) / y_test_rescaled)) * 100
    print(f"MSE: {mse_test:.4f}")
    print(f"MAE: {mae_test:.4f}")
    print(f"RMSE: {rmse_test:.4f}")
    print(f"R² Score: {r2_test:.4f}")
    print(f"MAPE: {mape_test:.2f}%")

    # Visualize test predictions
    plt.figure(figsize=(15, 8))
    plt.plot(y_test_rescaled, label='Actual', color='black', linewidth=2)
    plt.plot(test_predictions_rescaled, label='Predicted', color='blue', linewidth=2)
    plt.title(f'Actual vs Predicted Close Prices (Test Set) for {symbol}')
    plt.xlabel('Time Steps')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Save model and scaler with stock name
    model_save_path = f'lstm_models/{symbol}_model.h5'
    scaler_save_path = f'lstm_models/{symbol}_scaler.pkl'
    model.save(model_save_path)
    joblib.dump(scaler, scaler_save_path)
    print(f"Model and scaler saved successfully for {symbol} at {model_save_path} and {scaler_save_path}")

    # Future prediction (next day)
    last_sequence = close_scaled[-time_step:].reshape(1, time_step, 1)
    next_day_prediction = model.predict(last_sequence)
    next_day_prediction_rescaled = scaler.inverse_transform(next_day_prediction)[0, 0]
    current_price = data['Close'].iloc[-1]
    print(f"Current Price for {symbol}: ${current_price:.2f}")
    print(f"Predicted Next Day Close Price for {symbol}: ${next_day_prediction_rescaled:.2f}")
    change_percent = ((next_day_prediction_rescaled - current_price) / current_price) * 100
    print(f"Change for {symbol}: {change_percent:+.2f}%")

def main():
    # Dynamically load all CSV files from the Datasets directory
    datasets_dir = 'datasets'
    for filename in os.listdir(datasets_dir):
        if filename.endswith('.csv'):
            symbol = os.path.splitext(filename)[0]
            data_path = os.path.join(datasets_dir, filename)
            try:
                data = load_and_preprocess_data(data_path, symbol)
                if data is None:
                    print(f"Skipping {symbol} due to data loading issues.")
                    continue

                # Visualize initial data
                visualize_data(data, symbol)

                # Train and evaluate model
                train_and_evaluate_model(data, symbol)

            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()