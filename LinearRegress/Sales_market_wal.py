# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:17:58 2024

@author: Oghale Enwa
"""


import numpy as np 
import pandas as pd 
import seaborn as sns
sns.set()

data = pd.read_csv("Dummy_Data_HSS.csv")
raw_data = data.copy()

total_missing = data.isna().sum()
# print("Total missing values:", total_missing)
data=data.dropna()
duplicated_rows = data.duplicated().sum()
data.rename(columns={"Social Media": "Social_Media"}, inplace=True)

# # Create a box plot to visualize the relationship between Influencer and Sales
# plt.figure(figsize=(10, 6))
# # Set the color palette to "Set3"
# sns.set_palette("Set3")
# sns.boxplot(x='Influencer', y='Sales', data=data)
# plt.xlabel('Influencer')
# plt.ylabel('Sales')
# plt.title('Box Plot of Sales by Influencer')
# plt.xticks(rotation=45) 

# numeric_columns = ['TV', 'Radio', 'Social_Media', 'Sales']

# # Set the color palette to a consistent one (optional)
# sns.set_palette("Set3")

# # Create a pair plot for the numeric variables
# sns.pairplot(data=data[numeric_columns], height=2)
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Define the independent variables and perform individual simple linear regressions
independent_variables = ['TV', 'Radio', 'Social_Media']
results = []

for variable in independent_variables:
    formula = f"Sales ~ {variable}"
    model = ols(formula=formula, data=data).fit()
    results.append((f"Simple Linear Regression: {variable} vs. Sales", model))

# # Evaluate and print the results for each simple linear regression
# for model_name, model in results:
#     print(f"=== {model_name} ===")
#     print(model.summary())
#     print("\n")


X = data[['TV', 'Radio', 'Social_Media']]
y = data['Sales']

# Add a constant term to the independent variables (intercept)
X = sm.add_constant(X)

# Fit the multivariate linear regression model
model = sm.OLS(y, X).fit()

# Display the model summary
print(model.summary())





