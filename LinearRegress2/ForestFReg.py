# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:57:55 2024

@author: Oghale Enwa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

raw_data = pd.read_csv("forestfires.csv")
df = raw_data.copy()

#sns.scatterplot(data=df, x='temp', y='RH')
#plt.title('Scatter Plot of Temperature vs Relative Humidity')
#plt.xlabel('Temperature (°C)')
#plt.ylabel('Relative Humidity (%)')
#plt.show()

# Line plot
# For demonstration, let's plot the trend of FFMC index over the months
#sns.lineplot(data=df, x='month', y='FFMC', ci=None)  # ci=None removes confidence intervals
#plt.title('Trend of FFMC Index Over Months')
#plt.xlabel('Month')
#plt.ylabel('FFMC Index')
#plt.xticks(rotation=45)

#sns.lineplot(data=df, x='day', y='FFMC')  # ci=None removes confidence intervals
#plt.title('Trend of FFMC Index Over Days of the Week')
#plt.xlabel('Day')
#plt.ylabel('FFMC Index')
#plt.xticks(rotation=45)

#sns.lineplot(x='day', y='FFMC', data=df, color='blue', alpha=0.7)
#plt.title('Trend of FFMC Index Over Days of the Week')
#plt.xlabel('Day')
#plt.ylabel('FFMC Index')

#sns.lineplot(x='month', y='FFMC', data=df, color='blue', alpha=0.7)
#plt.title('Trend of FFMC Index Over Days of the Week')
#plt.xlabel('Month')
#plt.ylabel('FFMC Index')


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Encode categorical variables ('month' and 'day')
label_encoder = LabelEncoder()
df['month'] = label_encoder.fit_transform(df['month'])
df['day'] = label_encoder.fit_transform(df['day'])

# Split the data into features (X) and target variable (y)
X = df.drop(['area'], axis=1)  # Drop 'area' since it's the target variable
y = (df['area'] > 0).astype(int)  # Convert 'area' to binary indicating forest fire occurrence

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions
predictions = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)


#from sklearn.naive_bayes import GaussianNB

#Initializing Naives Bayes
#gnb = GaussianNB()

# Train the model
#gnb.fit(X_train, y_train)

# Make predictions
#predictions_gnb = gnb.predict(X_test)

# Evaluate the model
#accuracy = accuracy_score(y_test, predictions_gnb)
#print("Accuracy:", accuracy)

#from sklearn.neighbors import KNeighborsClassifier
#neigh = KNeighborsClassifier(n_neighbors=4)

# Train the model
#neigh.fit(X_train, y_train)

# Make predictions
#predictions_knn = neigh.predict(X_test)

# Evaluate the model
#accuracy = accuracy_score(y_test, predictions_knn)
#print("Accuracy:", accuracy)

# Define a range of values for k (number of neighbors)
#k_values = range(1, 21)  # Example: 1 to 20

# Initialize lists to store accuracy scores for different values of k
#accuracy_scores = []

# Iterate over each value of k
#for k in k_values:
    # Initialize and train the KNN classifier
#    knn_classifier = KNeighborsClassifier(n_neighbors=k)
#    knn_classifier.fit(X_train, y_train)
    
    # Make predictions on the test set
#    y_pred = knn_classifier.predict(X_test)
    
    # Calculate accuracy and store it
#    accuracy = accuracy_score(y_test, y_pred)
#    accuracy_scores.append(accuracy)

# Plot the graph
#plt.figure(figsize=(10, 6))
#plt.plot(k_values, accuracy_scores, marker='o', linestyle='-')
#plt.title('KNN Performance')
#plt.xlabel('Number of Neighbors (k)')
#plt.ylabel('Accuracy')
#plt.xticks(np.arange(min(k_values), max(k_values)+1, 1))  # Set x-axis ticks to integer values
#plt.grid(True)


















