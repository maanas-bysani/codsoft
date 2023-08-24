# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 12:35:11 2023

@author: Maanas
"""

#%% imports

#%%% importing libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#%%% importing data set

iris = sns.load_dataset("iris")
print(iris)
print(iris.describe())
column_headers = iris.columns.values.tolist()

#%% exploratory analysis

print("Target Labels", iris["species"].unique()) #this is the variable we are trying to predict
sns.scatterplot(x="sepal_width", y="sepal_length", data=iris, hue="species", palette='Set2') #relationship between sepal width and length
plt.show()

sns.scatterplot(x="petal_width", y="petal_length", data=iris, hue="species", palette='Set2') #relationship between petal width and length
plt.show()

#%% creating testing and training data

from sklearn.model_selection import train_test_split #to split data into training and testing datasets
x = iris.drop("species", axis=1) #all data except species name
y = iris["species"] #this is our y aka the target variable, i.e. what we are gonna predict

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#%%% create a dataframe using the training values

train_df = pd.DataFrame(x_train, columns=column_headers[:-1])
print(train_df)

#%%% create a dataframe using the testing values

test_df = pd.DataFrame(x_test) #create a dataframe using the testing values
test_df_index = test_df.index #extract index values of rows used

print(iris['species'].iloc[test_df_index]) #print species for the training data from the known dataset - useful to check correctness
print('these are the species names we are expecting')

#%% building and training model

from sklearn.neighbors import KNeighborsClassifier #this is the model we will use

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train) #training the model with the training data

#%% testing model with testing data

prediction = knn.predict(x_test)
test_df['Prediction']=prediction #adding a column to the testing dataframe created earlier; this appearance makes more sense than just the species name - this is rather arbitrary and holds no value

print(test_df['Prediction'])
print('these are the species names our model has predicted')

#%% checking number of non-matches, i.e. whether the prediction matches the expectation

checking_df = pd.DataFrame(iris['species'].iloc[test_df_index])
checking_df['prediction'] = prediction
print(checking_df)

non_match_count = len(checking_df.loc[checking_df.species != checking_df.prediction])
print("-------------------------------------------------------------------")
print("the number of predictions which dont match the expectations is:", non_match_count)

#%% testing model with random inputs

x_input = np.array([[5.4, 2.7, 5.1, 0.8]]) #inputs
prediction = knn.predict(x_input)
print("based on the input values, we expect it to be:", prediction)

#%% adding the random input data into a temporary dataframe - might be useful later

temp_df = iris
temp_df.loc[len(temp_df.index)] = [x_input[:,0], x_input[:,1], x_input[:,2], x_input[:,3], prediction] #adding values from input to temp_df dataframe
print(temp_df)
