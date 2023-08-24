# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 12:55:30 2023

@author: Maanas
"""

#%% imports
#%%% importing libraries

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import lazypredict
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

#%%% importing datasets

train_data = pd.read_csv('train.csv')
print(train_data)
test_data = pd.read_csv('test.csv') #this is a dataset without survival details - we will use this towards the end only; the model will be built based on the training dataset
print(test_data)

#%% replacing survived column with string instead of numbers to allow easier comprhension - this messes up code below, so not using

# train_data['Survived'] = train_data['Survived'].replace(to_replace=0, value='Victim')
# train_data['Survived'] = train_data['Survived'].replace(to_replace=1, value='Survivor')
# print(train_data)

#%% exploratory analysis

#%%% nice palettes

p1 = 'plasma'
p2 = ['#272483', '#1E78DC']
p3 = ['#45377B', '#DED02C']
p4 = ['#DC8F95', '#645C5D']
p5 = ['#3F7185', '#E1D7C3']

#%%% creating a new variable for grouping all data

Dummy = 'All Passengers'
train_data['All Passengers'] = Dummy

#%%% survival by sex

women = train_data.loc[train_data.Sex == 'female']["Survived"]
surv_women = sum(women)/len(women)

print('the percentage of women who survived is:', surv_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
surv_men = sum(men)/len(men)

print('the percentage of men who survived is:', surv_men)

gender = train_data[['Sex', 'Survived', 'All Passengers']]
surv_gender = gender[gender.Survived == 1]
labels= 'Women', 'Men'
plt.pie(surv_gender.value_counts(), labels=labels, colors=p3)
plt.show()


#%%% survival by age

age = train_data[['Age', 'Survived', 'All Passengers']]
surv_age = age[age.Survived == 1]
no_surv_age = age[age.Survived == 0]

sns.boxplot(x='Survived', y='Age', data=age, palette=p3)
plt.xlabel(None)
plt.show()

sns.boxplot(x='Survived', y='Age', data=train_data, hue='Sex', palette=p3)
plt.xlabel(None)
plt.show()

sns.stripplot(x='Survived', y='Age', data=train_data, hue='Sex', palette=p3)
plt.xlabel(None)
plt.show()

sns.stripplot(y='Age', data=age, hue='Survived', x='All Passengers', size=2, palette=p3)
plt.xlabel(None)
plt.show()

#%%% survival by fare

fare = train_data[['Fare', 'Survived', 'All Passengers']]
surv_fare = fare[fare.Survived == 1]
no_surv_fare = fare[fare.Survived == 0]

sns.stripplot(y='Fare', data=fare, hue='Survived', x='All Passengers', size=2, palette=p1)
plt.xlabel(None)
plt.show()

sns.stripplot(y='Fare', data=fare, hue='Survived', x='Survived', size = 2, palette=p1)
plt.xlabel(None)
plt.show()

sns.boxplot(y='Fare', data=fare, hue='Survived', x='Survived', linewidth=0.8, palette=p1)
plt.xlabel(None)
plt.show()


#%% survival by age group

#%%% total passenger by age group

temp_df=train_data
temp_df = train_data[train_data['Age'].notna()]
bins=[0,18,25,35,50,60,75,100] #lower and upper bounds included
labels = ['0-18', '19-25', '26-35', '36-50', '51-60', '61-75', '76+']
temp_df['Age Group']=pd.cut(temp_df['Age'],bins,labels=labels)
print(temp_df['Age Group'].value_counts())

sns.barplot(x=temp_df['Age Group'].value_counts().index, y=temp_df['Age Group'].value_counts())
plt.show()

#%%% creating a new dataframe with values for survived and non survived

age_group_survival_counts = temp_df.groupby(['Age Group', 'Survived']).size().unstack()
age_group_survival_counts = age_group_survival_counts.reset_index()
age_group_survival_counts_melted_df = pd.melt(age_group_survival_counts, id_vars=['Age Group'], value_vars=[0, 1], var_name='Survived', value_name='Count')

#%%% side by side plot of survived and non survived using pandas

age_group_survival_counts.plot(kind='bar', stacked=False, color=p2)
plt.xlabel('Age Group')
locs, xticks_labels = plt.xticks()
plt.xticks(locs, labels, rotation=0)
plt.ylabel('Count')
plt.title('Age Group Distribution by Survival')
plt.legend(title='Survived', labels=['Not Survived', 'Survived'])
plt.show()

#%%% stacked plot of survived and non survived using pandas

age_group_survival_counts.plot(kind='bar', stacked=True, color=p2)
plt.xlabel('Age Group')
locs, xticks_labels = plt.xticks()
plt.xticks(locs, labels, rotation=0)
plt.ylabel('Count')
plt.title('Age Group Distribution by Survival')
plt.legend(title='Survived', labels=['Not Survived', 'Survived'])
plt.show()

#%%% stacked plot of survived and non survived using seaborn

sns.countplot(data=temp_df, x='Age Group', hue='Survived', dodge=False, palette=p2)
plt.show()

#%%% side by side plot of survived and non survived using pandas

sns.countplot(data=temp_df, x='Age Group', hue='Survived', dodge=True, palette=p2)
plt.show()

#%% creating copy of df to use and alter 

model_data = train_data

#%% converting values from strings to floats for ML modeling

model_data['Sex'] = model_data['Sex'].replace(to_replace='female', value=0)
model_data['Sex'] = model_data['Sex'].replace(to_replace='male', value=1)
print(model_data)

print(model_data['Embarked'].value_counts())
model_data['Embarked'] = model_data['Embarked'].replace(to_replace='C', value=0)
model_data['Embarked'] = model_data['Embarked'].replace(to_replace='Q', value=1)
model_data['Embarked'] = model_data['Embarked'].replace(to_replace='S', value=2)
print(model_data)


#%% extracting only useful data for ML modeling 

features1 = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Survived']
model_data = model_data[features1]
print(model_data)

#%% removing all data with NaN points

model_data.dropna(axis = 0, how = 'any', inplace=True)
print(model_data)

#%% defining the label and target variables 

label_vars = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']

x = model_data[label_vars]
print(x)
y=model_data['Survived']
print(y)

#%% splitting the dataset into training and testing datasets

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)

#%% finding best model to use

#%%% classifier models comparision

from lazypredict.Supervised import LazyClassifier 
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)

clf_model, clf_prediction= clf.fit(x_train, x_test, y_train, y_test)
print(clf_model)
print("---------")
print(clf_prediction)

#%%% regression models comparision

from lazypredict.Supervised import LazyRegressor 
reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)

reg_model, reg_prediction= reg.fit(x_train, x_test, y_train, y_test)
print(reg_model)
print("---------")
print(reg_prediction)

#%% logistic regression

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
x_test['Survived'] = y_pred
print('prediction:', x_test)
checking_df = pd.DataFrame(y_test)
checking_df['Prediction'] = y_pred
non_match_count = len(checking_df.loc[checking_df.Survived != checking_df.Prediction])
print(non_match_count)
print("-------------------------------------------------------------------")
print('the result is :')
print("the number of predictions which dont match the expectations is:", non_match_count)
print('Accuracy : '+str(metrics.accuracy_score(y_test,y_pred)))
print('F1 score: '+str(metrics.f1_score(y_test,y_pred,average='macro')))

#%% ada boost classifier

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)
from sklearn.ensemble import AdaBoostClassifier
ab_clf=AdaBoostClassifier()
ab_clf.fit(x_train,y_train)
y_pred=ab_clf.predict(x_test)
x_test['Survived'] = y_pred
print('prediction:', x_test)
checking_df = pd.DataFrame(y_test)
checking_df['Prediction'] = y_pred
non_match_count = len(checking_df.loc[checking_df.Survived != checking_df.Prediction])
print(non_match_count)
print("-------------------------------------------------------------------")
print('the result is :')
print("the number of predictions which dont match the expectations is:", non_match_count)
print('Accuracy : '+str(metrics.accuracy_score(y_test,y_pred)))
print('F1 score: '+str(metrics.f1_score(y_test,y_pred,average='macro')))

#%% random forest classfier

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
x_test['Survived'] = y_pred
print('prediction:', x_test)
checking_df = pd.DataFrame(y_test)
checking_df['Prediction'] = y_pred
non_match_count = len(checking_df.loc[checking_df.Survived != checking_df.Prediction])
print(non_match_count)
print("-------------------------------------------------------------------")
print('the result is :')
print("the number of predictions which dont match the expectations is:", non_match_count)
print('Accuracy : '+str(metrics.accuracy_score(y_test,y_pred)))
print('F1 score: '+str(metrics.f1_score(y_test,y_pred,average='macro')))

#%%


















































#%%
'''
non_survived = age_group_survival_counts[0]
survived = age_group_survival_counts[1]
total = age_group_survival_counts[0] + age_group_survival_counts[1]

bar1 = sns.barplot(x='Age Group', y=total, data=age_group_survival_counts, color='darkblue', label='Total')
bar2 = sns.barplot(x='Age Group', y=survived, data=age_group_survival_counts, color='lightblue', label='Survived')
plt.legend()
plt.show()





#%%
grouped_surv_age = temp_df[temp_df.Survived == 1]
grouped_surv_age['Age Group']=pd.cut(grouped_surv_age['Age'],bins,labels=labels)
print(grouped_surv_age['Age Group'].value_counts())

sns.barplot(x=grouped_surv_age['Age Group'].value_counts().index, y=grouped_surv_age['Age Group'].value_counts())
plt.show()

grouped_no_surv_age = temp_df[temp_df.Survived == 0]
grouped_no_surv_age['Age Group']=pd.cut(grouped_no_surv_age['Age'],bins,labels=labels)
print(grouped_no_surv_age['Age Group'].value_counts())

sns.barplot(x=grouped_no_surv_age['Age Group'].value_counts().index, y=grouped_no_surv_age['Age Group'].value_counts())
plt.show()




#%%

x_train = train_data[features]
print(x_train)
y_train=train_data['Survived']
print(y_train)

x_test = test_data[features]
print(x_test)
# y_test=test_data['Survived']
# print(y_test)





temp_df = train_data[train_data['Age'].notna()]
# temp_df['Age'] = temp_df['Age'].dropna()
# temp_df['Age'] = temp_df['Age'].astype('int')

labels = ['0-18', '19-25', '26-35', '36-50', '51-60', '61-75', '76+']
bins=[-1,19,26,36,51,61,76,101]
binned_values = np.histogram(temp_df['Age'], bins=bins)[0].tolist() # use [0] to just get the counts
temp_df_hist = pd.DataFrame.from_dict(dict(zip(labels, binned_values)), orient='index').reset_index()
temp_df_hist.columns = ['ranges', 'counts']

sns.barplot(x='ranges', y='counts', data=temp_df_hist)
plt.show()


print(temp_df_hist)



temp_df['Age Group'] = pd.cut(temp_df['Age'], bins=bins, labels=labels)

# Count the occurrences of each age group for different survival statuses
age_group_survival_counts = temp_df.groupby(['Age Group', 'Survived']).size().unstack()

# Reset index for better visualization
age_group_survival_counts = age_group_survival_counts.reset_index()

# Plot using Seaborn with stacked bars
sns.barplot(data=age_group_survival_counts, x='Age Group', y=[0, 1], bottom=[0, 0])



sns.stripplot(data=train_data, x="Sex", y="Survived", hue="Pclass", kind="bar", palette=p2)
plt.show()



#%% removing all data with NaN points

x.dropna(axis = 0, how = 'any', inplace=True)
print(x)
y.dropna(axis = 0, how = 'any', inplace=True)
print(y)















'''
