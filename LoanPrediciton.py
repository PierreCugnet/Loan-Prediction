# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 09:48:41 2020

@author: Pierre Cugnet

Project Description: Personnal project
"""

import pandas as pd
import time
import re
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv("loan_dataset.csv")

#data exploration
df=df.drop('Loan_ID',axis=1)
df.describe()
df.head()
df.info()
df.dtypes

# Changing Credit_History to object type because 1 or 0
df['Credit_History'] = df['Credit_History'].astype('O') 



sns.set()
sns.countplot(df["Loan_Status"])
print('The percentage of Y class : %.2f' % (df['Loan_Status'].value_counts()[0] / len(df)))
print('The percentage of N class : %.2f' % (df['Loan_Status'].value_counts()[1] / len(df)))

#data visualization
sns.countplot(x="Married", hue="Loan_Status", data=df)
sns.countplot(x="Gender", hue="Loan_Status", data=df)
sns.countplot(x="Dependents", hue="Loan_Status", data=df)
sns.countplot(x="Education", hue="Loan_Status", data=df)
sns.countplot(x="Self_Employed", hue="Loan_Status", data=df)
sns.countplot(x="Property_Area", hue="Loan_Status", data=df)
sns.countplot(x="Credit_History", hue="Loan_Status", data=df)

plt.scatter(df["ApplicantIncome"], df["Loan_Status"]) #no pattern

#Missing values
df.isnull().sum().sort_values(ascending=False)
#Got a couple of missing values

cat_data=[]
num_data=[]

for i,c in enumerate(df.dtypes):
    print(i,c)
    if c == object:
        cat_data.append(df.iloc[:,i])
    else:
        num_data.append(df.iloc[:,i])
cat_data=pd.DataFrame(cat_data).transpose()
num_data=pd.DataFrame(num_data).transpose()
print(num_data)
print(cat_data)

print(num_data.isnull().sum())

#filling missing numerical values with their median
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan, strategy="median")
imputer.fit(num_data)
num_data=pd.DataFrame(imputer.transform(num_data), columns=num_data.columns)
print(num_data.isnull().sum())

#Replacing missing values of categorical data by their most frequent occurence
cat_data = cat_data.apply(lambda x:x.fillna(x.value_counts().index[0]))
print(cat_data.isnull().sum())

#No more missing values


#Encoding categorical data with LabelEncoder

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


#target column preprocessing
target_values = {'Y': 1 , 'N' : 0}
target = cat_data['Loan_Status']
cat_data.drop('Loan_Status', axis=1, inplace=True)
target = target.map(target_values)

#other columns
for i in cat_data:
    cat_data[i]=le.fit_transform(cat_data[i])
print(cat_data)

#Preprocessing done: Concatening the lists

df_preprocessed = pd.concat([cat_data, num_data, target], axis=1)

# Creating Train and Test set

X=df_preprocessed.iloc[:,:-1]
y=df_preprocessed[["Loan_Status"]]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


#Calculating Null Accuracy for comparison and better interpretation
null_acc = 1 - y_test.mean()
print(null_acc)

#Part 1: Logistic regression

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train, y_train)

y_pred_class=logreg.predict(X_test)

from sklearn import metrics
acc=metrics.accuracy_score(y_test, y_pred_class)
recall_score = metrics.recall_score(y_test, y_pred_class) #Sensitivity : When the value is 1 how often is he correct?
precision = metrics.precision_score(y_test, y_pred_class)
print(f'Accuracy = {acc}')
print(f'Recall Score = {recall_score}')
print(f'Precision = {precision}')


cm = metrics.confusion_matrix(y_test, y_pred_class)
tp = cm[1,1]
tn = cm[0,0]
fp = cm[0,1]
fn = cm[1,0]

