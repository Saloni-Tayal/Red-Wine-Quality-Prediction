# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 19:24:08 2019

@author: Sal
"""

import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score

ds = pd.read_csv('winequality-red.csv')
df = pd.DataFrame(ds)
df = shuffle(df)
df.columns

df['quality'].describe()

#count of each target variable
from collections import Counter
Counter(df['quality'])

#count of the target variable
sns.countplot(x='quality', data=df)

#Heatmap for correlation
#correlation matrix
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)


k = 10
cols = corrmat.nlargest(k, 'quality')['quality'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.20)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 12}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# Looking at the hm we can conclude Citric acid andfixed acidity we can choose one of them. 
#lets choose citric acid as it is more related to quality.

sns.pairplot(df)
# Still couldnt define any correlation between the variables

######################################################################################################
######################################################################################################
######################################################################################################

# use box and whisker plots for outliers

#Plot a boxplot to check for Outliers
#Target variable is Quality. So will plot a boxplot each column against target variable
sns.boxplot('quality', 'fixed acidity', data = df)
sns.boxplot('quality', 'citric acid', data = df)
sns.boxplot('quality', 'residual sugar', data = df)
sns.boxplot('quality', 'chlorides', data = df)
sns.boxplot('quality', 'free sulfur dioxide', data = df)
sns.boxplot('quality', 'total sulfur dioxide', data = df)
sns.boxplot('quality', 'density', data = df)
sns.boxplot('quality', 'pH', data = df)
sns.boxplot('quality', 'sulphates', data = df)
sns.boxplot('quality', 'alcohol', data = df)



#boxplots show many outliers for quite a few columns. Describe the dataset to get a better idea on what's happening
df.describe()
#fixed acidity - 25% - 7.1 and 50% - 7.9. Not much of a variance. Could explain the huge number of outliers
#volatile acididty - similar reasoning
#citric acid - seems to be somewhat uniformly distributed
#residual sugar - min - 0.9, max - 15!! Waaaaay too much difference. Could explain the outliers.
#chlorides - same as residual sugar. Min - 0.012, max - 0.611
#free sulfur dioxide, total suflur dioxide - same explanation as above

######################################################################################################
######################################################################################################
######################################################################################################

#next we shall create a new column called Review. This column will contain the values of 1,2, and 3. 
#1 - Bad
#2 - Average
#3 - Excellent
#This will be split in the following way. 
#1,2,3 --> Bad
#4,5,6,7 --> Average
#8,9,10 --> Excellent
#Create an empty list called Reviews
reviews = []
for i in df['quality']:
    if i >= 1 and i <= 3:
        reviews.append('1')
    elif i >= 4 and i <= 7:
        reviews.append('2')
    elif i >= 8 and i <= 10:
        reviews.append('3')
df['Reviews'] = reviews
######################################################################################################
######################################################################################################
######################################################################################################

#view final data
df.columns

df['Reviews'].unique()

Counter(df['Reviews'])

x = df.iloc[:,:11]
y = df['Reviews']


######################################################################################################
######################################################################################################
######################################################################################################


# MODELLING THE DATA
# Scale the data using Standard Scaler

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

#view the scaled features
print(x)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# Random Forest Classifier

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)

#Let's see how our model performed
print(classification_report(y_test, pred_rfc))

#Confusion matrix for the random forest classification
print(confusion_matrix(y_test, pred_rfc))


#  Stochastic Gradient Decent Classifier

sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)

print(classification_report(y_test, pred_sgd))

print(confusion_matrix(y_test, pred_sgd))


# Support Vector Classifier

svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)

print(classification_report(y_test, pred_svc))


######################################################################################################
######################################################################################################
######################################################################################################

# Increase the accuracy of our model






























