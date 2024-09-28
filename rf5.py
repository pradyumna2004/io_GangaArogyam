import gspread
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import time  # For creating delay

# Define the scope and credentials
scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(r"C:\Users\prady\Downloads\mlp-forecasting-project-1f2f855b3108.json", scope)

client = gspread.authorize(creds)

# Open the Google Sheet by URL
sheet_url = "https://docs.google.com/spreadsheets/d/1mLaEQiBuDV3A1on--e4O1HTrHu-Cp-n5QZOuUBy3Xcs/edit?usp=sharing"
sheet = client.open_by_url(sheet_url).sheet1

# Get all records from the sheet
data = sheet.get_all_records()

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Take the last 100 data points
df = df.tail(100)

# Define the parameters to forecast
parameters = df.columns[1:]  # Assumes the first column is YY-MM-DD//H-M-S

def prepare_data(df, parameter):
    X = np.arange(len(df)).reshape(-1, 1)  # Time index as feature
    y = df[parameter].astype(float).values  # Convert to float explicitly
    return X, y

def forecast_values_rf(X, y, num_forecasts, window_size):
    # Create the features for RandomForest using a sliding window approach
    X_train = np.array([y[i:i+window_size] for i in range(len(y)-window_size)])
    y_train = y[window_size:]
    
    # Standardize the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    
    # Train the RandomForest model
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train_scaled, y_train_scaled)
    
    # Generate forecasts
    forecasts = []
    last_window = y[-window_size:]
    for _ in range(num_forecasts):
        last_window_scaled = scaler_X.transform([last_window])
        next_forecast_scaled = rf.predict(last_window_scaled)
        next_forecast = scaler_y.inverse_transform(next_forecast_scaled.reshape(-1, 1)).ravel()[0]
        forecasts.append(next_forecast)
        last_window = np.append(last_window[1:], next_forecast)
    
    return np.array(forecasts)

num_forecasts = 50
window_size = 10  # Number of previous values to use for forecasting

# Open the new Google Sheet to store forecasted data
new_sheet_url = "https://docs.google.com/spreadsheets/d/1FxNXwMxwcjGFpoAmKOdc3qzlOgao5w5nJsU03A7d3MQ/edit?usp=sharing"
new_sheet = client.open_by_url(new_sheet_url).sheet1

# Prepare data for the new sheet
new_data = {'Time Index': list(range(len(df), len(df) + num_forecasts))}

# Initialize the plot
plt.figure(figsize=(15, 5 * len(parameters)))  # Adjusted size for clarity

# Prepare lists for the lines to update each parameter's plot
original_lines = []
forecast_lines = []

# Set up subplots for each parameter
for i, parameter in enumerate(parameters, 1):
    plt.subplot(len(parameters), 1, i)
    
    original_line, = plt.plot([], [], label=f'Original {parameter}', color='blue')
    forecast_line, = plt.plot([], [], label=f'Forecasted {parameter}', linestyle='--', color='red')
    
    plt.xlim(0, len(df) + num_forecasts)
    plt.ylim(df[parameter].min() - 5, df[parameter].max() + 5)
    plt.xlabel('Time Index')
    plt.legend()

    # Append the lines to the lists to update later
    original_lines.append(original_line)
    forecast_lines.append(forecast_line)

# Gradually plot all parameters at the same time
for j in range(len(df)):  # Plot original data gradually
    for i, parameter in enumerate(parameters):
        X, y = prepare_data(df, parameter)
        original_lines[i].set_data(range(j + 1), y[:j + 1])
    plt.pause(0.1)  # Pause to simulate gradual plotting

# Forecast and plot each parameter gradually
for j in range(num_forecasts):
    for i, parameter in enumerate(parameters):
        X, y = prepare_data(df, parameter)
        forecast = forecast_values_rf(X, y, num_forecasts, window_size)
        forecast_lines[i].set_data(range(len(y), len(y) + j + 1), forecast[:j + 1])
        if j == num_forecasts - 1:
            new_data[f'Forecasted {parameter}'] = forecast
    
    plt.pause(0.1)  # Pause to simulate gradual plotting

plt.tight_layout()
plt.show()

# Convert new_data dictionary to DataFrame
new_df = pd.DataFrame(new_data)

# Convert DataFrame to list of lists (required for gspread)
data_to_update = [new_df.columns.values.tolist()] + new_df.values.tolist()

# Ensure enough rows in the sheet for forecasted data
rows_needed = len(new_df) + 1  # +1 for the header row
current_rows = len(new_sheet.get_all_values())

if rows_needed > current_rows:
    new_sheet.add_rows(rows_needed - current_rows)

# Use append_rows() to avoid overwriting existing data
new_sheet.append_rows(data_to_update)