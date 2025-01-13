import pandas as pd
import os
input_directory='project/data'
file_path = os.path.join(input_directory, 'MSFT_merged_data.csv')
output_directory='project/clean_data'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def clean_and_analyze_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Drop duplicate columns by creating a DataFrame without duplicates
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Check for NaN values and calculate the percentage of NaNs in each column
    nan_info = pd.DataFrame(df.isna().sum(), columns=['NaN_count'])
    nan_info['NaN_percentage'] = (nan_info['NaN_count'] / len(df)) * 100
    
    # Add technical details about columns
    nan_info['Data_Type'] = df.dtypes.values  # Add data types
    nan_info['Mean'] = df.mean(numeric_only=True)  # Calculate mean for numeric columns
    nan_info['Median'] = df.median(numeric_only=True)  # Calculate median for numeric columns
    nan_info['Std_Dev'] = df.std(numeric_only=True)  # Calculate standard deviation for numeric columns
    
    # Ensure all columns are included, even those without NaNs
    nan_info['Column_Name'] = nan_info.index  # Add column names
    nan_info = nan_info.reset_index(drop=True)  # Reset index for cleaner output
    
    return df, nan_info

# Path to the merged CSV file
file_path = 'project/data/MSFT_merged_data.csv'

# Clean the data and get NaN information
cleaned_data, nan_summary = clean_and_analyze_data(file_path)
cleaned_data.to_csv(os.path.join(output_directory, 'cleaned_MSFT_data.csv'), index=False)
nan_summary.to_csv(os.path.join(output_directory, 'nan_summary_MSFT_data.csv'), index=False)

# Display the cleaned data and NaN summary
print("Cleaned Data:")
print(cleaned_data.head())
print("\nNaN Summary:")
print(nan_summary)
