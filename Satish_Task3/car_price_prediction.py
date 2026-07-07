# Import python Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Loading The Dataset
data = pd.read_csv("car data.csv")
print("")

# Display first 5 rows
print(data.head())

# CONVERTING TEXT COLUNM INTO NUMBER
data['Fuel_Type'] = data['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1, 'CNG': 2})
data['Selling_type'] = data['Selling_type'].map({'Dealer': 0, 'Individual': 1})
data['Transmission'] = data['Transmission'].map({'Manual': 0, 'Automatic': 1})

# CREATING A NEW FEATURE
data['Car_Age'] = 2026 - data['Year']

# SELECT INPUT AND OUTPUT COLUMNS
X = data[['Present_Price','Driven_kms','Fuel_Type',
          'Selling_type','Transmission','Owner','Car_Age']]

y = data['Selling_Price']

# SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# TRAINING MODEL
model = LinearRegression()
model.fit(X_train, y_train)

# PREDICTING PRICES
y_pred = model.predict(X_test)

# MODEL PERFORMANCE
score = model.score(X_test, y_test)
print("\nModel Accuracy (R2 Score):", round(score, 2))

# GRAPH
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)

# PERFECT PREDICTION LINE
plt.plot([y_test.min(),y_test.max()],
         [y_test.min(),y_test.max()])

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")

plt.show()