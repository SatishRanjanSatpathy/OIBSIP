# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
#data = pd.read_csv(r"C:\Users\satpa\internship\OIBSIP\Satish_Task2\Advertising.csv")
data = pd.read_csv("Advertising.csv")
# Display first 5 rows
print(data.head())

# Features and target
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict sales
predictions = model.predict(X_test)

# Compare actual and predicted values
result = pd.DataFrame({
    'Actual Sales': y_test,
    'Predicted Sales': predictions
})

print("\nSales Prediction:")
print(result.head())

# r2 score
r2 = r2_score(y_test, predictions)

print("R2 Score:", round(r2, 2))

# Graph
plt.scatter(y_test, predictions)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.show()