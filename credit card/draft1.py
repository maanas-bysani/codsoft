# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 18:14:43 2023

@author: Maanas
"""

#%% importing libraries
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import lazypredict

#%% importing data

data = pd.read_csv('creditcard.csv')
print(data.head())

#%% understanding the data

print(data.describe())
print(data.columns)

#%% checking for Nan values

print(data.isnull().sum().sum())
# print(data.isnull().sum())

#%% get ratio of fraud and non-fraud entries

print('not fraud', round((data['Class'].value_counts()[0]/len(data))*100,2))
print('fraud', round((data['Class'].value_counts()[1]/len(data))*100,2))

#%% nice color palettes for plotting

p1 = 'plasma'
p2 = ['#272483', '#1E78DC']
p3 = ['#45377B', '#DED02C']
p4 = ['#DC8F95', '#645C5D']
p5 = ['#3F7185', '#E1D7C3']

#%% plot of ratio of fraud and non-fraud entries

sns.countplot('Class', data=data, palette = p3, log=True) #log plot helps in visualisation
plt.title('Distribution of Fraudulent and Non-Fraudulent Transactions, log-scale')
plt.show()

sns.countplot('Class', data=data, palette = p3)
plt.title('Distribution of Fraudulent and Non-Fraudulent Transactions')
plt.show()

#%% distribution of transaction amount and transaction time

fig, ax = plt.subplots(1, 2, figsize=(20,5))

amount_val = data['Amount'].values
time_val = data['Time'].values

sns.distplot(amount_val, ax=ax[0], color=p4[0])
ax[0].set_title('Distribution of Transaction Amount')
# ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color=p4[1])
ax[1].set_title('Distribution of Transaction Time')
# ax[1].set_xlim([min(time_val), max(time_val)])

plt.show()

#%% transaction amount and transaction time are not scaled; all other columns are scaled. so scaling these 2 columns for consistency

from sklearn.preprocessing import StandardScaler, RobustScaler

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

amount_scaled = rob_scaler.fit_transform(data['Amount'].values.reshape(-1,1))
time_scaled = rob_scaler.fit_transform(data['Amount'].values.reshape(-1,1))

data.drop(['Time','Amount'], axis=1, inplace=True)
print(data.head())

#%% inserting in front as easier to read when printing

data.insert(0, 'amount_scaled', amount_scaled) 
data.insert(1, 'time_scaled', time_scaled)
print(data.head())

#%% seperating label and target variables

x = data.drop('Class', axis=1)
y = data['Class']

#%% splitting dataframe and creating training and testing dataframes

from sklearn.model_selection import train_test_split
original_x_train, original_x_test, original_y_train, original_y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#%% creating a new dataframe with equal no of fraud and non-fraud entries (chosen at random). this step exists because the dataset is greatly imbalanced

data = data.sample(frac=1)

fraud_data = data.loc[data['Class'] == 1]
non_fraud_data = data.loc[data['Class'] == 0][:len(fraud_data.index)]

equal_dist_data = pd.concat([fraud_data, non_fraud_data])
new_data = equal_dist_data.sample(frac=1)
print(new_data.head())

#%% proving the equal distribution of fraud and non-fraud entries in the new dataframe

sns.countplot('Class', data=new_data, palette = p3)
plt.title('Distribution of Fraudulent and Non-Fraudulent Transactions in New Dataset')
plt.show()

print('Propotion of Fraudulent and Non-Fraudulent Transactions in New Dataset: ')
print(new_data['Class'].value_counts()/len(new_data))

#%% nice colormaps for plotting

cmap1 = p1
cmap2 = 'RdYlBu'

#%% creating a correlation heatmap using both the original dataframe and the new dataframe

data_corr = data.corr()
plt.figure(figsize=(30,20))
sns.heatmap(data_corr, cmap = cmap2)
plt.title('Heatmap for Entire Dataset (dont use for model-fitting')
plt.show()

new_data_corr = new_data.corr()
plt.figure(figsize=(30,20))
sns.heatmap(new_data_corr, cmap = cmap2)
plt.title('Heatmap for Sample Dataset (use for model-fitting')
plt.show()

#%% finding columns that are most positively and most negatively correlated - these are the columns we should be exploring primarily

most_neg_corr = new_data_corr['Class'].sort_values(ascending=True).head(5)
most_pos_corr = new_data_corr['Class'].sort_values(ascending=False).head(6)
most_pos_corr = most_pos_corr.drop('Class') #dropping this due to self correlation

#%% box plots of most negatively correlated data

f, axes = plt.subplots(ncols=5, figsize=(30,5))

for i in range(0,len(most_neg_corr)):
    sns.boxplot(x='Class', y=most_neg_corr.index[i], data=new_data, palette=p5, ax=axes[i])
    axes[i].set_title(most_neg_corr.index[i] + ' vs Class (Negative Correlation)')

f.suptitle('Most Negatively Correlated Data')    
plt.show()

#%% box plots of most positively correlated data

f, axes = plt.subplots(ncols=5, figsize=(30,5))

for i in range(0,len(most_pos_corr)):
    sns.boxplot(x='Class', y=most_pos_corr.index[i], data=new_data, palette=p5, ax=axes[i])
    axes[i].set_title(most_pos_corr.index[i] + ' vs Class (Negative Correlation)')

f.suptitle('Most Positively Correlated Data')    
plt.show()

#%% splitting the new dataframe to create training and testing dataframes
# we will build our models using this dataframe but the final test will be made using the original dataframe!

X = new_data.drop('Class', axis=1)
Y = new_data['Class']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#%% getting accuracy of different classifier models with current dataset

from lazypredict.Supervised import LazyClassifier 
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)

clf_model, clf_prediction= clf.fit(x_train, x_test, y_train, y_test)
print(clf_model)
print("---------")
# print(clf_prediction)

#%% removing outliers from most negatively correlated factors - using multiple of IQR

new_data2 = new_data

for i in range(0,3):
    dynamic_variables = {}
    index_value = most_neg_corr.index[i]
    variable_name = index_value + '_fraud_data'
    # print(variable_name)
    dynamic_variables[variable_name] = new_data2[most_neg_corr.index[i]].loc[new_data2['Class']==1].values
    # print(dynamic_variables[variable_name])
    
    # outlier_fraud_data = new_data[most_neg_corr.index[0]].loc[new_data['Class']==1].values
    outlier_fraud_data = dynamic_variables[variable_name]
    lower_q = np.percentile(outlier_fraud_data, 25)
    upper_q = np.percentile(outlier_fraud_data, 75)
    median = np.median(outlier_fraud_data)
    iqr = upper_q - lower_q
    threshold = 1.5
    new_cut_off_iqr = iqr * threshold
    lower_cut_off = lower_q - new_cut_off_iqr
    # print(lower_cut_off)
    upper_cut_off = upper_q + new_cut_off_iqr
    # print(upper_cut_off)
    outliers = [x for x in outlier_fraud_data if x < lower_cut_off or x > upper_cut_off]
    print('the number of outliers in {0} are: {1}' .format(most_neg_corr.index[i], len(outliers)))
    # print('the outliers in {0} are: {1}' .format(most_neg_corr.index[i], outliers))
    new_data2 = new_data2.drop(new_data2[(new_data2[index_value] < lower_cut_off) | (new_data2[index_value] > upper_cut_off)].index)
    print(len(new_data2))

reduced_outlier_neg_corr_data = new_data2

#%% plotting box plots to check if outliers have indeed been removed

f, axes = plt.subplots(1,3, figsize=(20,5))

for i in range(0,3):
    sns.boxplot(x='Class', y=new_data[most_neg_corr.index[i]], data=new_data, palette=p5, ax=axes[i])
    axes[i].set_title(most_neg_corr.index[i] + ' vs Class (Negative Correlation)')
f.suptitle('Most Negatively Correlated Data - Original Data')    
plt.show()

f, axes = plt.subplots(1,3, figsize=(20,5))

for i in range(0,3):
    sns.boxplot(x='Class', y=reduced_outlier_neg_corr_data[most_neg_corr.index[i]], data=reduced_outlier_neg_corr_data, palette=p5, ax=axes[i])
    axes[i].set_title(most_neg_corr.index[i] + ' vs Class (Negative Correlation - Outlier Reduced Data)')
f.suptitle('Most Negatively Correlated Data- Outlier Reduced Data')
plt.show()

#%% removing outliers from most positively correlated factors - using multiple of IQR

new_data2 = new_data

for i in range(0,3):
    dynamic_variables = {}
    index_value = most_pos_corr.index[i]
    variable_name = index_value + '_fraud_data'
    # print(variable_name)
    dynamic_variables[variable_name] = new_data2[most_pos_corr.index[i]].loc[new_data2['Class']==1].values
    # print(dynamic_variables[variable_name])
    
    # outlier_fraud_data = new_data[most_pos_corr.index[0]].loc[new_data['Class']==1].values
    outlier_fraud_data = dynamic_variables[variable_name]
    lower_q = np.percentile(outlier_fraud_data, 25)
    upper_q = np.percentile(outlier_fraud_data, 75)
    median = np.median(outlier_fraud_data)
    iqr = upper_q - lower_q
    threshold = 1.5
    new_cut_off_iqr = iqr * threshold
    lower_cut_off = lower_q - new_cut_off_iqr
    # print(lower_cut_off)
    upper_cut_off = upper_q + new_cut_off_iqr
    # print(upper_cut_off)
    outliers = [x for x in outlier_fraud_data if x < lower_cut_off or x > upper_cut_off]
    print('the number of outliers in {0} are: {1}' .format(most_pos_corr.index[i], len(outliers)))
    # print('the outliers in {0} are: {1}' .format(most_pos_corr.index[i], outliers))
    new_data2 = new_data2.drop(new_data2[(new_data2[index_value] < lower_cut_off) | (new_data2[index_value] > upper_cut_off)].index)
    print(len(new_data2))

reduced_outlier_pos_corr_data = new_data2

#%% plotting box plots to check if outliers have indeed been removed

f, axes = plt.subplots(1,3, figsize=(20,5))

for i in range(0,3):
    sns.boxplot(x='Class', y=new_data[most_pos_corr.index[i]], data=new_data, palette=p5, ax=axes[i])
    axes[i].set_title(most_pos_corr.index[i] + ' vs Class (Positive Correlation)')
f.suptitle('Most Positively Correlated Data - Original Data')    
plt.show()

f, axes = plt.subplots(1,3, figsize=(20,5))

for i in range(0,3):
    sns.boxplot(x='Class', y=reduced_outlier_pos_corr_data[most_pos_corr.index[i]], data=reduced_outlier_pos_corr_data, palette=p5, ax=axes[i])
    axes[i].set_title(most_pos_corr.index[i] + ' vs Class (Positive Correlation)')
f.suptitle('Most Positvely Correlated Data- Outlier Reduced Data')
plt.show()

#%% removing anomalies from both: most pos and most neg factors - combing above steps together to create a new dataset with all outliers removed

outlier_reduced_data = new_data

for i in range(0,3):
    dynamic_variables = {}
    index_value = most_neg_corr.index[i]
    variable_name = index_value + '_fraud_data'
    # print(variable_name)
    dynamic_variables[variable_name] = outlier_reduced_data[most_neg_corr.index[i]].loc[outlier_reduced_data['Class']==1].values
    # print(dynamic_variables[variable_name])
    
    # outlier_fraud_data = new_data[most_neg_corr.index[0]].loc[new_data['Class']==1].values
    outlier_fraud_data = dynamic_variables[variable_name]
    lower_q = np.percentile(outlier_fraud_data, 25)
    upper_q = np.percentile(outlier_fraud_data, 75)
    median = np.median(outlier_fraud_data)
    iqr = upper_q - lower_q
    threshold = 1.5
    new_cut_off_iqr = iqr * threshold
    lower_cut_off = lower_q - new_cut_off_iqr
    # print(lower_cut_off)
    upper_cut_off = upper_q + new_cut_off_iqr
    # print(upper_cut_off)
    outliers = [x for x in outlier_fraud_data if x < lower_cut_off or x > upper_cut_off]
    print('the number of outliers in {0} are: {1}' .format(most_neg_corr.index[i], len(outliers)))
    # print('the outliers in {0} are: {1}' .format(most_neg_corr.index[i], outliers))
    outlier_reduced_data = outlier_reduced_data.drop(outlier_reduced_data[(outlier_reduced_data[index_value] < lower_cut_off) | (outlier_reduced_data[index_value] > upper_cut_off)].index)
    print(len(outlier_reduced_data))

for i in range(0,3):
    dynamic_variables = {}
    index_value = most_pos_corr.index[i]
    variable_name = index_value + '_fraud_data'
    # print(variable_name)
    dynamic_variables[variable_name] = outlier_reduced_data[most_pos_corr.index[i]].loc[outlier_reduced_data['Class']==1].values
    # print(dynamic_variables[variable_name])
    
    # outlier_fraud_data = new_data[most_pos_corr.index[0]].loc[new_data['Class']==1].values
    outlier_fraud_data = dynamic_variables[variable_name]
    lower_q = np.percentile(outlier_fraud_data, 25)
    upper_q = np.percentile(outlier_fraud_data, 75)
    median = np.median(outlier_fraud_data)
    iqr = upper_q - lower_q
    threshold = 1.5
    new_cut_off_iqr = iqr * threshold
    lower_cut_off = lower_q - new_cut_off_iqr
    # print(lower_cut_off)
    upper_cut_off = upper_q + new_cut_off_iqr
    # print(upper_cut_off)
    outliers = [x for x in outlier_fraud_data if x < lower_cut_off or x > upper_cut_off]
    print('the number of outliers in {0} are: {1}' .format(most_pos_corr.index[i], len(outliers)))
    # print('the outliers in {0} are: {1}' .format(most_pos_corr.index[i], outliers))
    outlier_reduced_data = outlier_reduced_data.drop(outlier_reduced_data[(outlier_reduced_data[index_value] < lower_cut_off) | (outlier_reduced_data[index_value] > upper_cut_off)].index)
    print(len(outlier_reduced_data))

#%% checking ratio of fraud and non-fraud entries after removing outliers 

sns.countplot('Class', data=outlier_reduced_data, palette = p3)
plt.title('Distribution of Fraudulent and Non-Fraudulent Transactions in New Dataset')
plt.show()

print('Propotion of Fraudulent and Non-Fraudulent Transactions in New Dataset: ')
print(outlier_reduced_data['Class'].value_counts()/len(outlier_reduced_data))

#%% plotting box plots to check if outliers have indeed been removed

f, axes = plt.subplots(1,3, figsize=(20,5))

for i in range(0,3):
    sns.boxplot(x='Class', y=outlier_reduced_data[most_neg_corr.index[i]], data=outlier_reduced_data, palette=p5, ax=axes[i])
    axes[i].set_title(most_neg_corr.index[i] + ' vs Class (Negative Correlation)')
f.suptitle('Most Negatively Correlated Data - Original Data')    
plt.show()

f, axes = plt.subplots(1,3, figsize=(20,5))

for i in range(0,3):
    sns.boxplot(x='Class', y=outlier_reduced_data[most_neg_corr.index[i]], data=outlier_reduced_data, palette=p5, ax=axes[i])
    axes[i].set_title(most_neg_corr.index[i] + ' vs Class (Negative Correlation - Outlier Reduced Data)')
f.suptitle('Most Negatively Correlated Data- Outlier Reduced Data')
plt.show()


f, axes = plt.subplots(1,3, figsize=(20,5))

for i in range(0,3):
    sns.boxplot(x='Class', y=outlier_reduced_data[most_pos_corr.index[i]], data=outlier_reduced_data, palette=p5, ax=axes[i])
    axes[i].set_title(most_pos_corr.index[i] + ' vs Class (Positive Correlation)')
f.suptitle('Most Positively Correlated Data - Original Data')    
plt.show()

f, axes = plt.subplots(1,3, figsize=(20,5))

for i in range(0,3):
    sns.boxplot(x='Class', y=outlier_reduced_data[most_pos_corr.index[i]], data=outlier_reduced_data, palette=p5, ax=axes[i])
    axes[i].set_title(most_pos_corr.index[i] + ' vs Class (Positive Correlation)')
f.suptitle('Most Positvely Correlated Data- Outlier Reduced Data')
plt.show()

#%% shorter name for the dataframe

model_data = outlier_reduced_data 

#%%

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import time
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report











#%%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections


'''
V14 = new_data['V14'].loc[new_data['Class'] == 1].values
lower_q, upper_q = np.percentile(V14, 25),  np.percentile(V14, 75)
print(lower_q)
print(upper_q)
iqr = upper_q - lower_q
print(iqr)
threshold = 1.5
cut_off = iqr*threshold
lower_cut_off, upper_cut_off = lower_q - cut_off, upper_q + cut_off
print(lower_cut_off)
print(upper_cut_off)

outliers = [x for x in V14 if x < lower_cut_off or x > upper_cut_off]
print(outliers)






#%%
print(most_pos_corr)

'''

