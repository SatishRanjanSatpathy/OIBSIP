# Iris Flower Classification Project

import pandas as pd  

from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score          


# loading dataset
iris_data = pd.read_csv(r"C:\Users\satpa\internship\OIBSIP\Satish_Task1\Iris.csv")
print("Dataset Loaded Successfully")
print()

# first 5 rows (head)
print(iris_data.head())

print()
print("Column Names")
print(iris_data.columns)

# separation of input and output Data
X = iris_data.drop("Species", axis=1) 
y = iris_data["Species"]

print()
print("Input Data")
print(X.head())

print()
print("Output Data")
print(y.head())

# splitting dataset into training and testing set
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
print()
print("Training Data Size :", Xtrain.shape)
print("Testing Data Size :", Xtest.shape)

# model creating
rf_model = RandomForestClassifier()

# training model
rf_model.fit(Xtrain, ytrain)

print()
print("Model Training Completed")

# prediction
prediction = rf_model.predict(Xtest)

print()
print("Predicted Output")
print(prediction)

# accuracy score
acc = accuracy_score(ytest, prediction)

print()
print("Accuracy")
print(acc)

