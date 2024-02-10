"""
FORMAT CSV DATA AS SQL/PYTHON-LEGIBLE
Third draft creating sql objects
"""
import csv, pandas as pd, matplotlib.pylab as plt

# Specify the path to your CSV file
import pandas

# Specify filepath -- Could be included in read_csv() function
csv_file_path = '/Users/justinmericle/Desktop/Data/Ai-HousingPrices/Stage 1/Housing.csv'

# read csv file as SQL data
data = pd.read_csv(csv_file_path)
print(data)
'''
# Mean Squared Loss Function manual calculation
def loss_function(m, b, points): # (slope, y-intercept, data points)
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].area # set x-axis to house 'area' values
        y = points.iloc[i].price # set y-axis to house 'price' values
        # use Mean Squared Error function to find TOTAL squared difference between predicted value and actual value
        total_error += (y - (m * x +b)) ** 2
    mean_squared_error = total_error / float(len(points)) # take average of total error to find average squared error
    return mean_squared_error
'''

def gradient_descent(m_now, b_now, points, learning_rate):
    m_gradient = 0
    b_gradient = 0

    n = len(points) # data length constant

    for i in range(n): # SUM(sigma) in derivative function
        x = points.iloc[i].area # set x-values to area of houses
        y = points.iloc[i].price # set y-values to prices of houses

        # use derivatives of Mean Squared Function to calculate gradient
        m_gradient += -(2/n) * x * (y - (m_now * x * b_now))
        b_gradient += -(2/n) * (y - (m_now * x * b_now))

    m = m_now - m_gradient * learning_rate
    b = b_now - b_gradient * learning_rate
    return m, b

m = 0
b = 0
learning_rate = 0.0001
epochs = 500 # number of iterations -- soaks up alot of memory


for i in range(epochs):
    if i % 50 == 0:
        print(f'Progress: {100 * (i / epochs)}%')
    m, b = gradient_descent(m, b, data, learning_rate)

print(m, b)

plt.scatter(data.area, data.price, color="black")
plt.plot(list(range(20, 80)), [m * x + b for x in range(20, 80)], color="red")
plt.show()


##### PLOT DATAPOINTS #####
## plt.scatter(data.area, data.price)
## plt.show()