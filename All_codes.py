import json
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch
import os

# Disable oneDNN custom operations warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load and preprocess data
def load_and_preprocess_data(test_files_count=2):
    json_files = glob.glob('data_json/*.json')
    data_list = []

    for file in json_files:
        with open(file, 'r') as f:
            data_list.append(json.load(f))

    # Split data into train and test based on file count
    train_data = data_list[:-test_files_count]
    test_data = data_list[-test_files_count:]

    # Prepare training data
    faults_train = []
    t_train_all = []
    p_train_all = []

    for data in train_data:
        for fault in data['faults']:
            for point in fault:
                faults_train.append(point)
        t_train_all.extend(data['t'])
        p_train_all.extend(data['p'])

    # Prepare testing data
    faults_test = []
    t_test_all = []
    p_test_all = []

    for data in test_data:
        for fault in data['faults']:
            for point in fault:
                faults_test.append(point)
        t_test_all.extend(data['t'])
        p_test_all.extend(data['p'])

    # Convert to numpy arrays
    t_train = np.array(t_train_all)
    p_train = np.array(p_train_all)
    t_test = np.array(t_test_all)
    p_test = np.array(p_test_all)

    # Normalize t and p to 0-1 range
    scaler_t = MinMaxScaler()
    scaler_p = MinMaxScaler()
    t_train = scaler_t.fit_transform(t_train.reshape(-1, 1)).flatten()
    p_train = scaler_p.fit_transform(p_train.reshape(-1, 1)).flatten()
    t_test = scaler_t.transform(t_test.reshape(-1, 1)).flatten()
    p_test = scaler_p.transform(p_test.reshape(-1, 1)).flatten()

    # Define a fixed sequence length (e.g., 100 time steps)
    sequence_length = 100

    # Create input sequences for training
    input_data_train = np.column_stack((t_train, p_train))
    X_train = []
    y_train = []

    for i in range(0, len(input_data_train) - sequence_length, sequence_length):
        X_train.append(input_data_train[i: i + sequence_length])
        y_train.append(faults_train[i // (sequence_length // 2) % len(faults_train)])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Create input sequences for testing
    input_data_test = np.column_stack((t_test, p_test))
    X_test = []
    y_test = []

    for i in range(0, len(input_data_test) - sequence_length, sequence_length):
        X_test.append(input_data_test[i: i + sequence_length])
        y_test.append(faults_test[i // (sequence_length // 2) % len(faults_test)])

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test, scaler_t, scaler_p

# Train model
def train_model(X_train, y_train):
    model = Sequential()
    model.add(Input(shape=(100, 2)))
    model.add(LSTM(100))
    model.add(Dense(2))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    
    # Save the model
    os.makedirs('trained_model', exist_ok=True)
    model.save('trained_model/best_model.keras')

# Load and test model
def test_model(X_test, y_test, scaler_t, scaler_p):
    model = load_model('trained_model/best_model.keras')
    predictions = model.predict(X_test)

    # Denormalize predictions
    predictions_denorm = scaler_t.inverse_transform(predictions.reshape(-1, 1)).reshape(-1, 2)

    # Compare predictions with ground truth
    plt.figure(figsize=(10, 6))
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    num_faults = len(y_test) // 2
    for i in range(num_faults):
        true_start = y_test[2*i]
        true_end = y_test[2*i + 1]
        pred_start = predictions_denorm[2*i]
        pred_end = predictions_denorm[2*i + 1]
        
        color = colors[i % len(colors)]
        plt.plot([true_start[0], true_end[0]], [true_start[1], true_end[1]], color + 'o-', label=f'True Fault {i+1}')
        plt.plot([pred_start[0], pred_end[0]], [pred_start[1], pred_end[1]], color + 'x--', label=f'Predicted Fault {i+1}')
    
    plt.legend()
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('True vs Predicted Fault Coordinates')
    plt.show()

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, predictions_denorm)
    print(f'Mean Squared Error: {mse}')

# Hyperparameter tuning function
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(100, 2)))
    model.add(layers.LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32)))
    model.add(layers.Dense(2))
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])),
                  loss='mean_squared_error')
    return model

# Hyperparameter tuning
def tune_model(X_train, y_train):
    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=3,
        directory='tuner_logs',
        project_name='fault_mapping'
    )

    tuner.search(X_train, y_train, epochs=50, validation_split=0.2)

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.summary()
    
    # Save the best model
    os.makedirs('trained_model', exist_ok=True)
    best_model.save('trained_model/best_tuned_model.keras')

def main():
    X_train, y_train, X_test, y_test, scaler_t, scaler_p = load_and_preprocess_data()

    # Uncomment the desired function to run
    train_model(X_train, y_train)
    # tune_model(X_train, y_train)
    test_model(X_test, y_test, scaler_t, scaler_p)

if __name__ == "__main__":
    main()
