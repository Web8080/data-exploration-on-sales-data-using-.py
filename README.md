Sales Forecasting Project

Overview

This project aims to analyze sales data and forecast future sales and revenues. Using historical sales data from retail centers, we employ various data analysis and machine learning techniques to estimate the impact of dropping products and predict sales and revenue for the upcoming year.

Features

Data Analysis: Analyze historical sales data to understand trends and patterns.
Sales Forecasting: Use time series forecasting models to predict future sales.
Impact Analysis: Evaluate the potential impact of dropping a product on total sales and revenue.
Dataset

The dataset used in this project is statsfinal.csv, which contains the following columns:

Date: Date of the sales record
Q-P1, Q-P2, Q-P3, Q-P4: Quantity of products sold
S-P1, S-P2, S-P3, S-P4: Sales revenue for each product
Installation

To run this project, you need to have Python installed along with several packages. You can install the required packages using pip:

bash
Copy code
pip install pandas numpy matplotlib statsmodels scikit-learn
Usage

Data Preparation:
Load the dataset and convert the Date column to datetime format.
Extract relevant features from the date.
Predict Sales:
Train a Linear Regression model to predict sales for specific dates.
Forecast sales and revenue for the year 2024 using an ARIMA model.
Analyze Product Impact:
Evaluate the impact of dropping one of the products on total sales and revenue.
Visualize the potential change in total sales.
Example Code
Here’s a brief example of how to use the code to predict sales for December 31st:

python
Copy code
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('path/to/statsfinal.csv')

# Preprocess and prepare data
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
data['DayOfYear'] = data['Date'].dt.dayofyear
data['Month'] = data['Date'].dt.month

X = data[['DayOfYear', 'Month']]
y = data['Q-P1']

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict sales for December 31st
X_pred = pd.DataFrame({'DayOfYear': [365], 'Month': [12]})
predicted_sales = model.predict(X_pred)
print(f"Estimated sales on December 31st: ${predicted_sales[0]:.2f}")
