# OIBSIP - DATA SCIENCE INTERNSHIP

## TASK 3: CAR PRICE PREDICTION WITH MACHINE LEARNING

### Introduction

This project aims to predict the selling price of used cars using Machine Learning. The model is trained on historical car data and estimates the selling price based on various features such as present price, kilometers driven, fuel type, transmission type, number of previous owners and car age.

### Python modules Used

* Pandas
* Matplotlib
* Scikit-learn

### Machine Learning Algorithm

* Linear Regression

### About the Dataset
* It consists of 301 records and following features related to each car.

### Features Used

* Present Price
* Driven Kilometers
* Fuel Type
* Selling Type
* Transmission
* Owner
* Car Age

### Workflow

1. Loaded the dataset using Pandas.
2. Converted categorical data into numerical values.
3. Created a new feature named **Car_Age**.
4. Selected input and output for training.
5. Split the dataset into training and testing sets.
6. Trained the model using Linear Regression.
7. Evaluated the model using the R² Score.
8. Visualized the results using a scatter plot with the perfect prediction line.

### Model Performance

The Linear Regression model achieved an **R² Score of 0.85**, which indicates that the model explains approximately **85% of the variance in car selling prices**.

### Conclusion

The project demonstrates how Linear Regression can be used to predict used car prices effectively. Feature engineering, such as adding car age helped improve the model's performance.
