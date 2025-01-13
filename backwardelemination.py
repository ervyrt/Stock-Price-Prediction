import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint
import seaborn as sns
output_directory = 'project/elemination_outputs'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
input_path= os.path.join('project', 'clean_data', 'cleaned_MSFT_data.csv')
output_path = os.path.join(output_directory, 'cleaned_MSFT_data.csv')
if not os.path.exists(output_path):
    with open(output_path, 'w') as file:
        file.write('')
df = pd.read_csv(input_path)
df['date'] = pd.to_datetime(df['date'])
column_names = df.columns.tolist()
print(column_names)
df.set_index('date', inplace=True)
from sklearn.metrics import mean_squared_error
from math import sqrt

target = '2. high'

df = df.astype(float)


def prepare_data(X, y):
    #to prevent ahead looking prejudice we shift y column 1row down
    y = df[target].shift(-1)
    X = df.drop(columns=target)

    X = X[:-1]  
    y = y[:-1]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    y_scaled = scaler_y.fit_transform(pd.DataFrame(y))
    y_scaled = pd.DataFrame(y_scaled)
    column_names = X_scaled.columns.tolist()
    print(column_names)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler_y

X_train, X_test, y_train, y_test,scaler_y = prepare_data(df, df[target])
#create checkpoint file
checkpoint_file = os.path.join(output_directory, 'checkpoint.json')  # Checkpoint file path
callback_path = os.path.join(output_directory, 'checkpoint.weights.h5')  # Change the extension to .weights.h5
callback = ModelCheckpoint(filepath=callback_path, save_weights_only=True, verbose=1)
# Backward elimination to retain the most important features; for start, I have chosen 375 features
def backward_elimination(X_train, y_train, X_test, y_test, test_size=0.2, random_state=42, min_features=1):
    """
    Perform backward elimination to retain the most important features.
    test_size: The percentage of the data to use for testing
    random_state: The random state to use for the train_test_split
    min_features: The number of columns we want to achieve
    """
    if X_train is None or y_train is None or X_test is None or y_test is None:
        logging.error("Input data is None. Cannot perform backward elimination.")
        return [], pd.DataFrame()  # Return empty DataFrame if input is None

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_file):
        checkpoint = pd.read_json(checkpoint_file)
        current_features = checkpoint['current_features'].tolist()
        elimination_scores_df = pd.read_json(checkpoint['scores'])
        logging.info("Loaded checkpoint.")
    else:
        current_features = list(X_train.columns)
        elimination_scores_df = pd.DataFrame(columns=['Num_Features', 'Eliminated_Feature', 'Score_MSE', 'Score_MAE', 'Score_RMSE', 'Score_MAPE', 'Score_SMAPE'])

    num_features = len(current_features)
    current_features_file = open(os.path.join(output_directory, 'current_features_BW.txt'), 'w')
    elimination = pd.DataFrame(columns=[
    'Num_Features', 
    'Eliminated_Feature', 
    'Score_MSE', 
    'Score_MAE', 
    'Score_RMSE', 
    'Score_MAPE', 
    'Score_SMAPE'
    ])
    elimination_scores_df.to_csv(os.path.join(output_directory, 'elimination_scores_BW.csv'), index=False, header=False, mode='a')
    while num_features > min_features:
        feature_scores = {}  # Dictionary to hold scores for each feature removal
        
        for feature in current_features:
            reduced_features = [f for f in current_features if f != feature]
            model = RandomForestRegressor(
                n_estimators=10,
                max_depth=8,
                random_state=random_state,
                n_jobs = 1
            )
            model.fit(X_train[reduced_features], y_train)
            num_features = len(reduced_features)
            predictions = model.predict(X_test[reduced_features])
            score_MSE = mean_squared_error(y_test, predictions)
            score_MAE = mean_absolute_error(y_test, predictions)
            score_RMSE = np.sqrt(score_MSE)  # RMSE is the square root of MSE
            score_MAPE = mean_absolute_percentage_error(y_test, predictions)
            score_SMAPE = mean_absolute_percentage_error(y_test, predictions)
            feature_scores[feature] = score_MSE  # Store the score for this feature removal
            
            # Append the scores to the elimination scores DataFrame
            new_row = pd.DataFrame({
                'Num_Features': [num_features],
                'Eliminated_Feature': [feature],
                'Score_MSE': [score_MSE],
                'Score_MAE': [score_MAE],
                'Score_RMSE': [score_RMSE],
                'Score_MAPE': [score_MAPE],
                'Score_SMAPE': [score_SMAPE]
            })
            new_row.to_csv(os.path.join(output_directory, 'elimination_scores_BW.csv'), index=False, header=False, mode='a')
            elimination_scores_df = pd.concat([elimination_scores_df, new_row], ignore_index=True)   
        # Identify the feature with the best score (lowest MSE)
        worst_score = min(feature_scores.values())  # Get the worst score to eliminate
        worst_features = [feature for feature, score in feature_scores.items() if score == worst_score] 
        for feature in worst_features:
            current_features.remove(feature)  # Remove all worst features
        num_features = len(current_features)
        current_features_file.write(f"{num_features}: {', '.join(current_features)}\n")  # Update the added feature file
        current_features_file.flush()  # Make sure the changes are written to the file
        logging.info(f"Eliminated features: {', '.join(worst_features)}. Remaining features: {num_features}")

        # Save checkpoint
        checkpoint_data = {
            'current_features': current_features,
            'scores': elimination_scores_df.to_json()
        }
        pd.Series(checkpoint_data).to_json(checkpoint_file)

    return current_features, elimination_scores_df  # Return the final set of features and the scores DataFrame

def create_scatterplot(data, output_path):
    # Creating histograms for all columns in relation to 'Num_Features'
    numeric_columns = [col for col in data.select_dtypes(include=['float64', 'int64']).columns if col != 'Num_Features']
    # Creating scatter plots for each numeric column against Num_Features
    plt.figure(figsize=(16, len(numeric_columns) * 5))
    for i, column in enumerate(numeric_columns, start=1):
        plt.subplot(len(numeric_columns), 1, i)
        
        # Scatter plot using Num_Features as x-axis and the corresponding column as y-axis
        plt.scatter(data['Num_Features'], data[column], alpha=0.7, edgecolor='k')
        plt.title(f'Scatter Plot of Num_Features vs {column}')
        plt.xlabel('Num_Features')
        plt.ylabel(column)
        plt.xticks(ticks=sorted(data['Num_Features'].unique()), rotation=45)
        
        # Adjusting y-axis to focus on smaller changes
        y_min, y_max = data[column].min(), data[column].max()
        plt.ylim(y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1)

    plt.tight_layout()

    plt.savefig(output_path, format='png', dpi=300)
    plt.close()
def create_boxplot(df, output_folder):
    # Set seaborn style for better aesthetics
    sns.set(style="whitegrid")

    # Adjust figure size for readability
    plt.figure(figsize=(12, 6))

    # Group by 'Num_Features' and plot 'Score_MAE'
    sns.boxplot(
        x='Num_Features', 
        y='Score_MAE', 
        data=df, 
        showfliers=False,  # Remove outliers to reduce clutter
        palette='coolwarm'  # Add a color palette for visual appeal
    )

    # Rotate x-axis labels for clarity
    plt.xticks(rotation=45)

    # Set plot title and labels
    plt.title('Improved Box Plot of error_MAE by num_features', fontsize=14)
    plt.xlabel('Number of Features', fontsize=12)
    plt.ylabel('Error MAE', fontsize=12)

    # Limit the number of ticks on x-axis
    x_ticks = df['Num_Features'].unique()[::10]  # Show every 10th tick
    plt.xticks(ticks=range(0, len(df['Num_Features'].unique()), 10), labels=x_ticks)

    # Save the improved plot
    output_path = os.path.join(output_folder, 'improved_boxplot_error_MAE.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()

    print(f"Improved boxplot saved to: {output_path}")

def save_eleminated_features(scores):
    change_column_names = {'67': 'Number of Features', 'Unnamed: 0': 'Eliminated Feature', '0.0002490866956281078': 'MSE', '0.0035978114530298643': 'MAE', '0.015782480655084224': 'RMSE', '5298059003.411585': 'MAPE', '5298059003.411585.1': 'SMAPE'}
    scores = scores.rename(columns=change_column_names)
    print(scores.columns)
    scores = scores.sort_values(by='MSE', ascending=True)
    best_combination = scores['Number of Features'].iloc[0]
    best_score = scores['MSE'].iloc[0]
    print(f"The best score is {best_score} for {best_combination} features")
    best_combination = int(best_combination)
    # Read the current features file to get the features corresponding to the best combination
    with open(os.path.join(output_directory, 'current_features_BW.txt'), 'r') as file:
        lines = file.readlines()
        # Extract the row corresponding to best_combination
        best_combination_row = lines[-best_combination+1]
        best_combination_row = best_combination_row.replace(f'{best_combination}: ', '')  # best_combination is the index
        print(best_combination_row)

        best_combination_row = best_combination_row.strip().split(',')
        for i in range(len(best_combination_row)):
            best_combination_row[i] = best_combination_row[i].strip() 
        best_combination_row += ['2. high'] 
        print(best_combination_row)
    best_combination_row.append('date')
    eleminated_data = df[best_combination_row]
    eleminated_data.to_csv(os.path.join(output_directory, 'eleminated_data.csv'), index=False)

    print(best_combination)
'''scores.to_csv(os.path.join(input_directory, 'elimination_scores_BW_sorted.csv'), index=False)
print(scores.columns)'''
if __name__ == "__main__":
 

    # Assuming 'timestamp_column' is the name of your timestamp column
    timestamp_column = 'Timestamp'  # Replace with your actual timestamp column name

    current_features, scores_df = backward_elimination(X_train, y_train,X_test, y_test, test_size=0.2, random_state=42, min_features=1)

    create_boxplot(scores_df, output_directory)

