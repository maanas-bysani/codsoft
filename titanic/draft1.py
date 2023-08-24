# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 17:47:38 2023

@author: Maanas
"""

#%% importing libraries

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#%%

data = pd.read_csv('train.csv') #training data
print(data.head())

test_data = pd.read_csv('test.csv') #testing data
print(test_data.head())

#%%
Dummy = 'All Passengers'
data['All Passengers'] = Dummy

#%%
women = data.loc[data.Sex == 'female']["Survived"]
surv_women = sum(women)/len(women)

print('the percentage of women who survived is:', surv_women)

men = data.loc[data.Sex == 'male']["Survived"]
surv_men = sum(men)/len(men)

print('the percentage of men who survived is:', surv_men)

#%%

gender = data[['Sex', 'Survived', 'All Passengers']]
surv_gender = gender[gender.Survived == 1]
labels= 'Women', 'Men'
plt.pie(surv_gender.value_counts(), labels=labels)
plt.show()

#%%
age = data[['Age', 'Survived', 'All Passengers']]
surv_age = age[age.Survived == 1]
no_surv_age = age[age.Survived == 0]

#%%

sns.boxplot(x='Survived', y='Age', data=age)
plt.show()

sns.boxplot(x='Survived', y='Age', data=data, hue='Sex')
plt.show()

sns.stripplot(x='Survived', y='Age', data=data, hue='Sex')
plt.show()

sns.stripplot(y='Age', data=age, hue='Survived', x='All Passengers', size=2)
plt.xlabel(None)
plt.show()


#%%

sns.stripplot(data=data, x="Sex", y="Survived", hue="Pclass", kind="bar")

#%%


fare = data[['Fare', 'Survived', 'All Passengers']]
surv_fare = fare[fare.Survived == 1]
no_surv_fare = fare[fare.Survived == 0]

sns.stripplot(y='Fare', data=fare, hue='Survived', x='All Passengers', size=2)
plt.xlabel(None)
plt.show()

#%%

sns.stripplot(y='Fare', data=fare, hue='Survived', x='Survived', size = 2)
plt.show()

#%%
sns.boxplot(y='Fare', data=fare, hue='Survived', x='Survived')
plt.show()



#%%
from lazypredict.Supervised import LazyClassifier
x = np.array(data.drop(["All Passengers", "Survived"], axis=1))
y = np.array(data['Survived'])

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30, random_state=42)


#%%

clf = LazyClassifier(verbose=0,ignore_warnings=True,custom_metric=None)
models, predictions = clf.fit(xtrain, xtest, ytrain, ytest)
print(models)

#%%
from lazypredict.Supervised import LazyRegressor

reg = LazyRegressor(verbose=0,ignore_warnings=False,custom_metric=None)
models, predictions = reg.fit(xtrain, xtest, ytrain, ytest)
print(models)

