import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from math import sqrt
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
output_directory = 'project/outputs'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
input_path= 'project/elemination_outputs/eleminated_data.csv'
df = pd.read_csv(input_path)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
column_names = df.columns.tolist()
print(column_names)

target = '2. high'

df = df.astype(float)

'''
for preventing ahead prejudice shifts data, 
scaling data and creating sequential data,
then spliting into '''
def prepare_data(X, y, time_steps=10):
    #to prevent ahead looking prejudice we shift y column 1row down
    y = df[target].shift(-1)
    X = df.drop(columns=target)

    X = X[:-1]  
    y = y[:-1]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled)
    y_scaled = scaler_y.fit_transform(pd.DataFrame(y))
    y_scaled = pd.DataFrame(y_scaled)
    Xs, ys = [], []
    for i in range(len(X_scaled) - time_steps):
        v = X_scaled.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y_scaled.iloc[i + time_steps].values)
    # Display the shape of X for better understanding
    print(f"Shape of X: {np.array(Xs).shape}")
    # Split the data
    X_seq= np.array(Xs)
    y_seq= np.array(ys)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler_y

X_train, X_test, y_train, y_test,scaler_y = prepare_data(df, df[target])
unit = list(best_params.get('units'))
optimizer = best_params.get('optimizer')
learning_rate = best_params.get('learning_rate')
batch_size = best_params.get('batch_size')
def create_lstm_model(units=unit, optimizer=optimizer, learning_rate=learning_rate):
    # Initialize the RNN
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])), 
        LSTM(units[0], return_sequences=True),
        LSTM(units[1], return_sequences=True),
        LSTM(units[2], return_sequences=False),
        Dense(1)
    ])
        # Dynamically setting the optimizer with the learning_rate
    if optimizer == 'Adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'RMSprop':
        opt = RMSprop(learning_rate=learning_rate)
    elif optimizer == 'Nadam':
        opt = Nadam(learning_rate=learning_rate)
    #optimizer = tf.keras.optimizers.get(optimizer)(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_absolute_percentage_error', 'mean_squared_error'])
    return model

# Define the path to save the best model
best_model_path = 'project/outputs/best_lstm_model.h5'

# Create a ModelCheckpoint callback to save the best model
checkpoint = ModelCheckpoint(best_model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
'''
# Fit the model with the checkpoint callback
model = create_lstm_model(units=unit, optimizer=optimizer, learning_rate=learning_rate)
history = model.fit(X_train, y_train, epochs=50, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2, callbacks=[checkpoint])

# Save the training history to a CSV file
history_df = pd.DataFrame(history.history)  # Convert history to DataFrame
history_df.to_csv('project/outputs/model_training_history.csv', index=False)  # Save to CSV
'''
# Load the best model
model = load_model(best_model_path)

# Make predictions

predicted_high_prices = model.predict(X_test)
predicted_high_prices = np.reshape(predicted_high_prices, (-1, 1))
predicted_high_prices = scaler_y.inverse_transform(predicted_high_prices)

actual_high_prices = np.reshape(y_test, (-1, 1))
actual_high_prices = scaler_y.inverse_transform(actual_high_prices)




# Create a DataFrame to save the predicted and actual prices
results_df = pd.DataFrame({
    'Actual High Prices': actual_high_prices.flatten(),
    'Predicted High Prices': predicted_high_prices.flatten()
})

# Save the results to a CSV file
results_df.to_csv(os.path.join(output_directory, 'predicted_vs_actual_high_pricescsv'), index=False)

# Optionally, plot predictions against actual values
plt.figure(figsize=(100, 40))
plt.plot(actual_high_prices, label='Actual High')
plt.plot(predicted_high_prices, label='Predicted High')
plt.title('High Price Prediction')
plt.xlabel('Time')
plt.ylabel('High Price')
plt.legend()
plt.savefig(os.path.join(output_directory, 'high_price_prediction.png'))



rmse = sqrt(mean_squared_error(actual_high_prices, predicted_high_prices))
print('Test RMSE: %.3f' % rmse)
