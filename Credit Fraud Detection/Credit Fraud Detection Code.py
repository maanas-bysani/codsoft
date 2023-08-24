# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 18:14:43 2023

@author: Maanas
"""

#%% imports and housekeeping

#%%% importing libraries
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#%%% importing data

data = pd.read_csv('creditcard.csv')
print(data.head())

#%%% nice color palettes and cmaps for plotting

p1 = 'plasma'
p2 = ['#272483', '#1E78DC']
p3 = ['#45377B', '#DED02C']
p4 = ['#DC8F95', '#645C5D']
p5 = ['#3F7185', '#E1D7C3']

cmap1 = p1
cmap2 = 'RdYlBu'

#%% pre processing the data

#%%% understanding the data

print(data.describe())
print(data.columns)

#%%% checking for Nan values

print(data.isnull().sum().sum())
# print(data.isnull().sum())

#%% exploratory analysis

#%%% get ratio of fraud and non-fraud entries

print('not fraud', round((data['Class'].value_counts()[0]/len(data))*100,2))
print('fraud', round((data['Class'].value_counts()[1]/len(data))*100,2))

#%%% plot of ratio of fraud and non-fraud entries

sns.countplot('Class', data=data, palette = p3, log=True) #log plot helps in visualisation
plt.title('Distribution of Fraudulent and Non-Fraudulent Transactions, log-scale')
plt.show()

sns.countplot('Class', data=data, palette = p3)
plt.title('Distribution of Fraudulent and Non-Fraudulent Transactions')
plt.show()

#%%% distribution of transaction amount and transaction time

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

#%% more data processing - scaling time and amount factors

#%%% transaction amount and transaction time are not scaled; all other columns are scaled. so scaling these 2 columns for consistency

from sklearn.preprocessing import StandardScaler, RobustScaler

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

amount_scaled = rob_scaler.fit_transform(data['Amount'].values.reshape(-1,1))
time_scaled = rob_scaler.fit_transform(data['Amount'].values.reshape(-1,1))

data.drop(['Time','Amount'], axis=1, inplace=True)
print(data.head())

#%%% inserting in front as easier to read when printing

data.insert(0, 'amount_scaled', amount_scaled) 
data.insert(1, 'time_scaled', time_scaled)
print(data.head())

#%% model building

#%%% seperating label and target variables

x = data.drop('Class', axis=1)
y = data['Class']

#%%% splitting dataframe and creating training and testing dataframes

from sklearn.model_selection import train_test_split
original_x_train, original_x_test, original_y_train, original_y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#%% creating a new dataframe with equal no of fraud and non-fraud entries (chosen at random). this step exists because the dataset is greatly imbalanced

data = data.sample(frac=1)

fraud_data = data.loc[data['Class'] == 1]
non_fraud_data = data.loc[data['Class'] == 0][:len(fraud_data.index)]

equal_dist_data = pd.concat([fraud_data, non_fraud_data])
new_data = equal_dist_data.sample(frac=1)
print(new_data.head())

#%%% proving the equal distribution of fraud and non-fraud entries in the new dataframe

sns.countplot('Class', data=new_data, palette = p3)
plt.title('Distribution of Fraudulent and Non-Fraudulent Transactions in New Dataset')
plt.show()

print('Propotion of Fraudulent and Non-Fraudulent Transactions in New Dataset: ')
print(new_data['Class'].value_counts()/len(new_data))

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

#%% finding most +vely and -vely correlated factors and understanding them

#%%% finding columns that are most positively and most negatively correlated - these are the columns we should be exploring primarily

most_neg_corr = new_data_corr['Class'].sort_values(ascending=True).head(5)
most_pos_corr = new_data_corr['Class'].sort_values(ascending=False).head(6)
most_pos_corr = most_pos_corr.drop('Class') #dropping this due to self correlation

#%%% box plots of most negatively correlated data

f, axes = plt.subplots(ncols=5, figsize=(30,5))

for i in range(0,len(most_neg_corr)):
    sns.boxplot(x='Class', y=most_neg_corr.index[i], data=new_data, palette=p5, ax=axes[i])
    axes[i].set_title(most_neg_corr.index[i] + ' vs Class (Negative Correlation)')

f.suptitle('Most Negatively Correlated Data')    
plt.show()

#%%% box plots of most positively correlated data

f, axes = plt.subplots(ncols=5, figsize=(30,5))

for i in range(0,len(most_pos_corr)):
    sns.boxplot(x='Class', y=most_pos_corr.index[i], data=new_data, palette=p5, ax=axes[i])
    axes[i].set_title(most_pos_corr.index[i] + ' vs Class (Negative Correlation)')

f.suptitle('Most Positively Correlated Data')    
plt.show()

#%% exploratory ML

#%%% splitting the new dataframe to create training and testing dataframes
# we will build our models using this dataframe but the final test will be made using the original dataframe!

X = new_data.drop('Class', axis=1)
Y = new_data['Class']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#%%% getting accuracy of different classifier models with current dataset

from lazypredict.Supervised import LazyClassifier 
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)

clf_model, clf_prediction= clf.fit(x_train, x_test, y_train, y_test)
print(clf_model)
print("---------")
# print(clf_prediction)

#%% bit more processing/cleaning - removing outliers, etc

#%%% removing outliers from most negatively correlated factors - using multiple of IQR

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

#%%% plotting box plots to check if outliers have indeed been removed

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

#%%% removing outliers from most positively correlated factors - using multiple of IQR

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

#%%% plotting box plots to check if outliers have indeed been removed

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

#%%% removing anomalies from both: most pos and most neg factors
#combing above steps together to create a new dataset with all outliers removed

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

#%%% checking ratio of fraud and non-fraud entries after removing outliers 

sns.countplot('Class', data=outlier_reduced_data, palette = p3)
plt.title('Distribution of Fraudulent and Non-Fraudulent Transactions in New Dataset')
plt.show()

print('Propotion of Fraudulent and Non-Fraudulent Transactions in New Dataset: ')
print(outlier_reduced_data['Class'].value_counts()/len(outlier_reduced_data))

#%%% plotting box plots to check if outliers have indeed been removed

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

#%% done with outlier reduction, now some more processing and model building

#%%% shorter name for the dataframe
# we will use this dataframe instead of the new_data dataframe to build and train our model

model_data = outlier_reduced_data 

#%%% reducing the dimensions of the dataset - this allows for quicker processing
# we will use the t-SNE algorithm which converts the dataset into a set of clusters
# we will also consider the PCA and SVD algorithms here

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import time
from sklearn.model_selection import cross_val_score


X = model_data.drop('Class', axis=1)
Y = model_data['Class']

start_time = time.time()
X_tsne = TSNE().fit_transform(X.values)
end_time = time.time()
print("T-SNE took %.2g s"%(end_time - start_time))

start_time = time.time()
X_pca = PCA().fit_transform(X.values)
end_time = time.time()
print("PCA took %.2g s"%(end_time - start_time))

start_time = time.time()
X_tsvd = TruncatedSVD().fit_transform(X.values)
end_time = time.time()
print("Truncated SVD took %.2g s" %(end_time - start_time))

#%%% plotting graphs of the 3 dimension-reduction algorithms

f, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(30,6))

sns.scatterplot(x=X_tsne[:,0],y= X_tsne[:,1], hue=Y, cmap='coolwarm', legend='auto', ax=ax1)
ax1.set_title('t-SNE', fontsize=14)
sns.scatterplot(x=X_pca[:,0],y= X_pca[:,1], hue=Y, cmap='coolwarm', legend='auto', ax=ax2)
ax2.set_title('PCA', fontsize=14)
sns.scatterplot(x=X_tsvd[:,0],y= X_tsvd[:,1], hue=Y, cmap='coolwarm', legend='auto', ax=ax3)
ax3.set_title('Truncated SVD', fontsize=14)

ax1.legend(['Fraud', 'No Fraud'])
ax2.legend(['Fraud', 'No Fraud'])
ax3.legend(['Fraud', 'No Fraud'])

f.suptitle('Dimensionality Reduction Output Clusters')

plt.show()

#%%finding best model

#%%% finding best classifier algorithm using the lazy predict library

X = model_data.drop('Class', axis=1)
Y = model_data['Class']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#%%% getting accuracy of different classifier models with current dataset

from lazypredict.Supervised import LazyClassifier 
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)

clf_model, clf_prediction= clf.fit(x_train, x_test, y_train, y_test)
print(clf_model)
print("---------")
# print(clf_prediction)

#%% model building

#%%% we will use the following classifier algorithms, selected based on scores and time taken

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier

classifiers = {"RandomForestClassifier": RandomForestClassifier(), "LogisticRegression": LogisticRegression(),"KNeighborsClassifier": KNeighborsClassifier(), "ExtraTreesClassifier": ExtraTreesClassifier(), "Support Vector Classifier":SVC()}

for key, classifier in classifiers.items():
    start_time = time.time()
    classifier.fit(x_train,y_train)
    score = cross_val_score(classifier, x_train,y_train, cv=10)
    end_time = time.time()
    print("{0} has a score of {1} and took {2} s".format(key,round(score.mean(),2),round(end_time - start_time,2)))
    
#%%% use gridsearchcv to find best parameters

from sklearn.model_selection import GridSearchCV

#LogisticRegression
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(x_train, y_train)
log_reg = grid_log_reg.best_estimator_

#KNeighborsClassifier
knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
grid_knears.fit(x_train, y_train)
knears_neighbors = grid_knears.best_estimator_

#Support Vector Classifier
svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid_svc = GridSearchCV(SVC(), svc_params)
grid_svc.fit(x_train, y_train)
svc = grid_svc.best_estimator_

# DecisionTree Classifier
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), "min_samples_leaf": list(range(5,7,1))}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(x_train, y_train)
tree_clf = grid_tree.best_estimator_

#%%% overfitting case and score

log_reg_score = cross_val_score(log_reg, x_train, y_train)
print("Log.Reg.Overfitting Case Cross Validation Score: {0} " .format(round(log_reg_score.mean(),4)))

knears_score = cross_val_score(knears_neighbors, x_train, y_train)
print("Log.Reg.Overfitting Case Cross Validation Score: {0} " .format(round(knears_score.mean(),4)))

svc_score = cross_val_score(svc, x_train, y_train)
print("Log.Reg.Overfitting Case Cross Validation Score: {0} " .format(round(svc_score.mean(),4)))

tree_score = cross_val_score(tree_clf, x_train, y_train)
print("Log.Reg.Overfitting Case Cross Validation Score: {0} " .format(round(tree_score.mean(),4)))


#%%% predict y_train values using the cross_val_predict function

from sklearn.model_selection import cross_val_predict

log_reg_pred = cross_val_predict(log_reg, x_train, y_train, cv=5, method="decision_function")

knears_pred = cross_val_predict(knears_neighbors, x_train, y_train, cv=5)

svc_pred = cross_val_predict(svc, x_train, y_train, cv=5, method="decision_function")

tree_pred = cross_val_predict(tree_clf, x_train, y_train, cv=5)

#%%% get roc score - area under the roc graph
from sklearn.metrics import roc_auc_score, roc_curve

print('Logistic Regression: ', round(roc_auc_score(y_train, log_reg_pred),2))

print('KNears Neighbors: ', round(roc_auc_score(y_train, knears_pred),2))

print('Support Vector Classifier: ', round(roc_auc_score(y_train, svc_pred),2))

print('Decision Tree Classifier: ', round(roc_auc_score(y_train, tree_pred),2))

#%%% plot the roc curves

log_reg_fpr, log_reg_tpr, log_reg_thresh = roc_curve(y_train, log_reg_pred)
knears_fpr, knears_tpr, knears_thresh = roc_curve(y_train, knears_pred)
svc_fpr, svc_tpr, svc_thresh = roc_curve(y_train, svc_pred)
tree_fpr, tree_tpr, tree_thresh = roc_curve(y_train, tree_pred)

plt.plot(log_reg_fpr, log_reg_tpr, label='Log Reg Score :{}'.format(round(roc_auc_score(y_train, log_reg_pred),2)))
plt.plot(knears_fpr, knears_tpr, label='Knears Score :{}'.format(round(roc_auc_score(y_train, knears_pred),2)))
plt.plot(svc_fpr, svc_tpr, label='SVC Score :{}'.format(round(roc_auc_score(y_train, svc_pred),2)))
plt.plot(tree_fpr, tree_tpr, label='Dec. Tree Score :{}'.format(round(roc_auc_score(y_train, tree_pred),2)))
plt.plot([0,1],[0,1], 'k--')
plt.xlabel('False Pos Rate')
plt.ylabel('True Pos Rate')
plt.title('ROC Curve')
plt.legend()
plt.annotate('Minimum ROC Score of \n50% to Achieve', xy=(0.5, 0.5), xytext=(0.61, 0.35), arrowprops=dict(facecolor='grey', shrink=0.05))
plt.show()

#%% further analysing log reg model

#%%% log reg roc curve
plt.plot(log_reg_fpr, log_reg_tpr, label='Log Reg Score :{}'.format(round(roc_auc_score(y_train, log_reg_pred),2)))
plt.plot([0,1],[0,1], 'k--')
plt.xlabel('False Pos Rate')
plt.ylabel('True Pos Rate')
plt.title('Log Reg ROC Curve')
plt.legend()
plt.show()

#%%% get precision and recall values;

#precision = tp / (tp + fp)
#recall: tp / (tp + fn)

from sklearn.metrics import precision_recall_curve

lr_precision, lr_recall, lr_threshold = precision_recall_curve(y_train, log_reg_pred)

#%% further testing of the model using smaller dataset

#%%% testing model using x_test data and creating a confusion matrix to analyse model effectiveness

from sklearn.metrics import confusion_matrix

log_reg = grid_log_reg.best_estimator_
log_reg.fit(x_train, y_train)
y_pred_log_reg = log_reg.predict(x_test)
log_reg_cf = confusion_matrix(y_test, y_pred_log_reg)


knears = grid_knears.best_estimator_
knears.fit(x_train, y_train)
y_pred_knears = knears.predict(x_test)
knears_cf = confusion_matrix(y_test, y_pred_knears)


svc = grid_svc.best_estimator_
svc.fit(x_train, y_train)
y_pred_svc = svc.predict(x_test)
svc_cf = confusion_matrix(y_test, y_pred_svc)


tree = grid_tree.best_estimator_
tree.fit(x_train, y_train)
y_pred_tree = tree.predict(x_test)
tree_cf = confusion_matrix(y_test, y_pred_tree)

#%%% plotting confusion matrix
# top right is true pos, top left is false pos, bottom left is true neg, bottom right is false neg
#we want high scores in true pos and true neg compared to the others

fig, ax = plt.subplots(2, 2,figsize=(30,20))


sns.heatmap(log_reg_cf, ax=ax[0][0], annot=True, cmap='copper')
ax[0, 0].set_title("Logistic Regression", fontsize=14)
ax[0, 0].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[0, 0].set_yticklabels(['', ''], fontsize=14, rotation=360)

sns.heatmap(knears_cf, ax=ax[0][1], annot=True, cmap='copper')
ax[0][1].set_title("KNearsNeighbors", fontsize=14)
ax[0][1].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[0][1].set_yticklabels(['', ''], fontsize=14, rotation=360)

sns.heatmap(svc_cf, ax=ax[1][0], annot=True, cmap='copper')
ax[1][0].set_title("Suppor Vector Classifier", fontsize=14)
ax[1][0].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[1][0].set_yticklabels(['', ''], fontsize=14, rotation=360)

sns.heatmap(tree_cf, ax=ax[1][1], annot=True, cmap='copper')
ax[1][1].set_title("Decision Tree Classifier", fontsize=14)
ax[1][1].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[1][1].set_yticklabels(['', ''], fontsize=14, rotation=360)

plt.suptitle('Confusion Matrices \n(Top Right -> True Pos, Top Left -> False Pos, Bottom Right -> True Neg, Bottom Left -> False Neg)', fontsize=25)

plt.show()

#%%% printing classification reports for each of the classification models

from sklearn.metrics import classification_report

print('Log Reg Classification Report:')
print(classification_report(y_test, y_pred_log_reg))
print("--"*50)

print('KNN Classification Report:')
print(classification_report(y_test, y_pred_knears))
print("--"*50)

print('SVC Classification Report:')
print(classification_report(y_test, y_pred_svc))
print("--"*50)

print('Dec. Tree Classification Report:')
print(classification_report(y_test, y_pred_tree))
print("--"*50)

#%% final testing using original dataset

#%%% repeat model testing with original x_test - this is from the huge, imbalanced dataset - expect results to be poor but non-zero

log_reg_org = grid_log_reg.best_estimator_
log_reg_org.fit(x_train, y_train)
y_pred_log_reg_org = log_reg_org.predict(original_x_test)
log_reg_cf_org = confusion_matrix(original_y_test, y_pred_log_reg_org)


knears_org = grid_knears.best_estimator_
knears_org.fit(x_train, y_train)
y_pred_knears_org = knears_org.predict(original_x_test)
knears_cf_org = confusion_matrix(original_y_test, y_pred_knears_org)


svc_org = grid_svc.best_estimator_
svc_org.fit(x_train, y_train)
y_pred_svc_org = svc_org.predict(original_x_test)
svc_cf_org = confusion_matrix(original_y_test, y_pred_svc_org)


tree_org = grid_tree.best_estimator_
tree_org.fit(x_train, y_train)
y_pred_tree_org = tree_org.predict(original_x_test)
tree_cf_org = confusion_matrix(original_y_test, y_pred_tree_org)

#%%% plotting confusion matrix

fig, ax = plt.subplots(2, 2,figsize=(30,20))


sns.heatmap(log_reg_cf_org, ax=ax[0][0], annot=True, cmap='copper')
ax[0, 0].set_title("Logistic Regression", fontsize=14)
ax[0, 0].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[0, 0].set_yticklabels(['', ''], fontsize=14, rotation=360)

sns.heatmap(knears_cf_org, ax=ax[0][1], annot=True, cmap='copper')
ax[0][1].set_title("KNearsNeighbors", fontsize=14)
ax[0][1].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[0][1].set_yticklabels(['', ''], fontsize=14, rotation=360)

sns.heatmap(svc_cf_org, ax=ax[1][0], annot=True, cmap='copper')
ax[1][0].set_title("Suppor Vector Classifier", fontsize=14)
ax[1][0].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[1][0].set_yticklabels(['', ''], fontsize=14, rotation=360)

sns.heatmap(tree_cf_org, ax=ax[1][1], annot=True, cmap='copper')
ax[1][1].set_title("Decision Tree Classifier", fontsize=14)
ax[1][1].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[1][1].set_yticklabels(['', ''], fontsize=14, rotation=360)

plt.suptitle('Confusion Matrices for predictions with the Imbalanced Dataset \n(Top Right -> True Pos, Top Left -> False Pos, Bottom Right -> True Neg, Bottom Left -> False Neg)', fontsize=25)

plt.show()

#%%% printing classification reports for each of the classification models

print('Log Reg Classification Report:')
print(classification_report(original_y_test, y_pred_log_reg_org))
print("--"*50)

print('KNN Classification Report:')
print(classification_report(original_y_test, y_pred_knears_org))
print("--"*50)

print('SVC Classification Report:')
print(classification_report(original_y_test, y_pred_svc_org))
print("--"*50)

print('Dec. Tree Classification Report:')
print(classification_report(original_y_test, y_pred_tree_org))
print("--"*50)

