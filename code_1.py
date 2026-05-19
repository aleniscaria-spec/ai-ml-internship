import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read CSV
data = pd.read_csv("Load_Extension.csv")

# Remove unwanted spaces from column names
data.columns = data.columns.str.strip()

# Select data
x = data[['Load']]
y = data['Extension']

# Create model
model = LinearRegression()
model.fit(x, y)

# Get coefficients
slope = model.coef_[0]
intercept = model.intercept_

print("Slope:", slope)
print("Intercept:", intercept)

# Equation
print(f"Regression Equation: y = {intercept:.4f} + {slope:.4f}x")

# Preditiction for 55 N
predicted = model.predict(pd.DataFrame({'Load':[55]}))

print("Predicted extension for 55 N:", predicted[0])

# Prediction
y_pred = model.predict(x)

# Plot of graph
plt.scatter(x, y, label='Observed Data')
plt.plot(x, y_pred, label='Regression Line')

plt.xlabel("Load (N)")
plt.ylabel("Extension (mm)")
plt.title("Load vs Spring Extension")

plt.legend()
plt.grid(True)

plt.show()