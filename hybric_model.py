import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
# Load the original cleaned data
cleaned_data = pd.read_csv('project/clean_data/cleaned_MSFT_data.csv')

# Retrieve the Date column
dates = cleaned_data['date']

grid_search_outputs_hybrid= 'project/grid_search_outputs_multilayer/grid_search_results.csv'
grid_search_results = pd.read_csv(grid_search_outputs_hybrid)
sorted_results = grid_search_results.sort_values(by='Mean Test Score', ascending=False)
best_params = eval(sorted_results.iloc[0]['Parameters'])
best_params= dict(best_params)
best_score = sorted_results.iloc[0]['Mean Test Score']

output_directory = 'project/outputs_hybrid'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
input_path= 'project/elemination_outputs/eleminated_data.csv'
df = pd.read_csv(input_path)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
column_names = df.columns.tolist()

target = '2. high'



'''
for preventing ahead prejudice shifts data, 
scaling data and creating sequential data,
then spliting into '''
def prepare_data(X, y, time_steps=60):
    adjusted_dates = dates.iloc[time_steps:].reset_index(drop=True)

    #to prevent ahead looking prejudice we shift y column 1row down
    y = df[target].shift(-1)
    X = df.drop(columns=target)

    X = X[:-1] 
    y = y[:-1]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled)  # Retain the original index
    y_scaled = scaler_y.fit_transform(pd.DataFrame(y))
    y_scaled = pd.DataFrame(y_scaled)
    Xs, ys = [], []
    for i in range(len(X_scaled) - time_steps):
        v = X_scaled.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y_scaled.iloc[i + time_steps].values)
    # Display the shape of X for better understanding
    print(f"Shape of X: {np.array(Xs).shape}")
    # Split the data ensuring the last rows are used for testing
    X_seq = np.array(Xs)
    y_seq = np.array(ys)
    split_index = int(len(X_seq) * 0.8)  # 80% for training
    X_train, X_test = X_seq[:split_index], X_seq[split_index:]
    y_train, y_test = y_seq[:split_index], y_seq[split_index:]
    test_dates = adjusted_dates.iloc[:y_test.shape[0]].reset_index(drop=True)
    return X_train, X_test, y_train, y_test, scaler_y, test_dates
units= best_params['units']
batch_size= best_params['batch_size']
learning_rate= best_params['learning_rate']
optimizer= best_params['optimizer']
X_train, X_test, y_train, y_test,scaler_y, test_dates = prepare_data(df, df[target])
model = Sequential([
    Conv1D(filters=units[0], kernel_size=2, activation='relu', input_shape=(60, 42)),
    MaxPooling1D(pool_size=2),
    LSTM(units[1], return_sequences=True),
    LSTM(units[2],return_sequences=True),
    LSTM(units[3]),
    Dense(1)
])

model.compile(optimizer=optimizer, loss='mse',  metrics=['mean_absolute_percentage_error','root_mean_squared_error'])

history = model.fit(X_train, y_train, epochs=50, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2)


# Save the training history to a CSV file
history_df = pd.DataFrame(history.history)  # Convert history to DataFrame
history_df.to_csv('project/outputs_hybrid/model_training_history_23_35.csv', index=False)  # Save to CSV

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
results_df['Date'] = test_dates
# Save the results to a CSV file
results_df.to_csv(os.path.join(output_directory, 'predicted_vs_actual_high_prices2331.csv'), index=False)

# Optionally, plot predictions against actual values
plt.figure(figsize=(256, 128))
plt.plot(actual_high_prices, label='Actual High')
plt.plot(predicted_high_prices, label='Predicted High')
plt.title('High Price Prediction')
plt.xlabel('Time')
plt.ylabel('High Price')
plt.legend()
plt.savefig(os.path.join(output_directory, 'high_price_prediction2331.png'))



rmse = sqrt(mean_squared_error(actual_high_prices, predicted_high_prices))
print('Test RMSE: %.3f' % rmse)
