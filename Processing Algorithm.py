'''
This processes the financial data available on 10-q and 10-k reports
'''
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Sample dataset with financial input factors and stock prices
data = {
    'EPS': [5.2, 4.8, 6.0, 3.5, 4.2],
    'P/E Ratio': [15.2, 14.5, 16.8, 12.3, 13.7],
    'Profit Margins': [0.10, 0.08, 0.12, 0.06, 0.09],
    'Debt-to-Equity Ratio': [0.5, 0.4, 0.6, 0.3, 0.5],
    'Stock Price': [50, 48, 55, 42, 47]
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data)

# Separate the features (X) and target variable (y)
X = df[['EPS', 'P/E Ratio', 'Profit Margins', 'Debt-to-Equity Ratio']]
y = df['Stock Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot predicted vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Stock Price')
plt.ylabel('Predicted Stock Price')
plt.title('Linear Regression: Actual vs Predicted Stock Price')
plt.show()
