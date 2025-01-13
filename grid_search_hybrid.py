import pandas as pd
import os
import numpy as np
import tensorflow as tf
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout
output_folder = 'project/grid_search_outputs_multilayer'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
input_path= 'project/elemination_outputs/eleminated_data.csv'
df = pd.read_csv(input_path)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
column_names = df.columns.tolist()
print(column_names)
from sklearn.metrics import mean_squared_error
from math import sqrt

target = '2. high'

df = df.astype(float)
'''
for preventing ahead prejudice shifts data, 
scaling data and creating sequential data,
then spliting into '''
def prepare_data(X, y, time_steps=60):
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
    print(f"Shape of X: {np.array(ys).shape}")
    # Split the data
    X_seq= np.array(Xs)
    y_seq = np.array(ys).reshape(-1, 1) 
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler_y

X_train, X_test, y_train, y_test,scaler_y = prepare_data(df, df[target])
'''
Model builder function
'''
print(X_train.shape, y_train.shape)
def create_hybrid_model(units, optimizer='Adam', learning_rate=0.001):
    model = Sequential([
        Input(shape=(60, 42)),
        Conv1D(filters=units[0], kernel_size=2, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.1),
        LSTM(units[1], return_sequences=True),
        LSTM(units[2], return_sequences=True),
        LSTM(units[3]),
        Dense(1)
    ])
    # Dynamically setting the optimizer with the learning_rate
    if optimizer == 'Adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'RMSprop':
        opt = RMSprop(learning_rate=learning_rate)
    elif optimizer == 'Nadam':
        opt = Nadam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_absolute_percentage_error', 'mean_squared_error'])
    return model


model = KerasRegressor(model=create_hybrid_model, verbose=0, learning_rate=0.001, units=128)


OPTIMIZER = ['Adam', 'Nadam']
BATCH_SIZE = [128, 256] 
LEARNING_RATE = [0.001, 0.01, 0.1, 0.5]
NODES = [[256, 128,64,32], [128, 64, 32, 16], [64, 32, 16, 8]]
# Update the param_grid to include neurons, layers, and learning rates
Param_grid = dict(optimizer=OPTIMIZER, units=NODES,
                  batch_size= BATCH_SIZE, learning_rate=LEARNING_RATE)

#Perform grid search
grid = GridSearchCV(estimator=model, param_grid=Param_grid, n_jobs=1, cv=2, verbose=3, scoring='neg_mean_absolute_percentage_error')
grid_result = grid.fit(X_train, y_train,  epochs=10)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
summary_df = pd.DataFrame(grid_result.cv_results_)
summary_df.to_csv(os.path.join(output_folder, "grid_search_summary.csv"), index=False)


# Create a DataFrame to store all results
results_df = pd.DataFrame({
    'Mean Test Score': means,
    'Standard Deviation': stds,
    'Parameters': params
})

# Save the results to a CSV file
results_csv_path = os.path.join(output_folder, "grid_search_results.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"Grid search results saved to '{results_csv_path}'.")

# Print all results
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



