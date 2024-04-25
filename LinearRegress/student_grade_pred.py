# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:03:50 2024

@author: Oghale Enwa
"""

import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = pd.read_csv("student-mat.csv")
raw_data = data.copy()
# print(data.isnull().sum())

# plt.figure(figsize=(12, 6))
# sns.histplot(data['G3'], bins=20, kde=True, color='skyblue')
# plt.title('Distribution of Final Grades (G3)')
# plt.xlabel('Grade')
# plt.ylabel('Frequency')

# # Distribution of grades based on study time
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='studytime', y='G3', data=data, palette='viridis')
# plt.title('Final Grade (G3) vs. Study Time')
# plt.xlabel('Weekly Study Time')
# plt.ylabel('Grade')

# #If school play any role in Grades?
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='school', y='G3', data=data, palette='viridis')
# plt.title('Final Grade (G3) Distribution by School')
# plt.xlabel('School')
# plt.ylabel('Grade')

# # Distribution of grades based on family relationship
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='famrel', y='G3', data=data, palette='viridis')
# plt.title('Final Grade (G3) vs. Family Relationship')
# plt.xlabel('Family Relationship')
# plt.ylabel('Grade')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score



X = data.drop(['G3'], axis=1)
y = data['G3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the column transformer for encoding categorical variables
categorical_features = X.select_dtypes(include=['object']).columns
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_features)], remainder='passthrough')

# Create a pipeline with the column transformer and linear regression model
pipeline = Pipeline(steps=[('preprocessor', ct), ('regressor', LinearRegression())])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Drop non-numeric columns and the target variable 'G3' for logistic regression
X = data.drop(['G3', 'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian'], axis=1)
y = (data['G3'] >= 10).astype(int)  # Convert grades to binary for logistic regression

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the column transformer for encoding categorical variables
categorical_features = X.select_dtypes(include=['object']).columns
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_features)], remainder='passthrough')

# Create a pipeline with the column transformer and logistic regression model
pipeline = Pipeline(steps=[('preprocessor', ct), ('classifier', LogisticRegression())])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")


