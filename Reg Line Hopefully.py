'''
Regression Algorithm, using deltas in previous quarters to calculate Stock Price
*** NEEDS WORK - WON'T PLOT REGRESSION LINE ***
'''

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Sample dataset with financial input factors and stock prices for two quarters
data_previous_quarter = {
    'EPS': [4.8, 4.5, 5.8, 3.2, 4.0],
    'P/E Ratio': [14.5, 14.0, 16.5, 11.8, 13.0],
    'Profit Margins': [0.09, 0.07, 0.11, 0.05, 0.08],
    'Debt-to-Equity Ratio': [0.4, 0.3, 0.5, 0.2, 0.4],
    'Stock Price': [48, 45, 54, 41, 46]
}

data_recent_quarter = {
    'EPS': [5.2, 4.8, 6.0, 3.5, 4.2],
    'P/E Ratio': [15.2, 14.5, 16.8, 12.3, 13.7],
    'Profit Margins': [0.10, 0.08, 0.12, 0.06, 0.09],
    'Debt-to-Equity Ratio': [0.5, 0.4, 0.6, 0.3, 0.5],
    'Stock Price': [50, 48, 55, 42, 47]
}

# Create DataFrames from the sample data for two quarters
df_previous_quarter = pd.DataFrame(data_previous_quarter)
df_recent_quarter = pd.DataFrame(data_recent_quarter)

# Calculate deltas between two quarters
df_delta = df_recent_quarter - df_previous_quarter

# Separate the features (X) and target variable (y) using recent quarter's stock prices
X_delta = df_delta[['EPS', 'P/E Ratio', 'Profit Margins', 'Debt-to-Equity Ratio']]
y_recent_quarter = df_recent_quarter['Stock Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_delta, y_recent_quarter, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot predicted vs actual values with regression line
plt.scatter(y_test, y_pred, label='Actual vs Predicted')
plt.plot(y_test, y_pred, 'o', color='red', label='Regression Line')
plt.xlabel('Actual Stock Price (Most Recent Quarter)')
plt.ylabel('Predicted Stock Price (Most Recent Quarter)')
plt.title('Linear Regression: Actual vs Predicted Stock Price (Most Recent Quarter)')
plt.legend()
plt.show()
