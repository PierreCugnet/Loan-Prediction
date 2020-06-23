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

#Import csv file
df=pd.read_csv("loan_dataset.csv")

# quick data exploration
df=df.drop('Loan_ID',axis=1)
df.describe()
df.head()
df.info()
df.dtypes

# Changing Credit_History to object type because 1 or 0
df['Credit_History'] = df['Credit_History'].astype('O') 


"""
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
"""
#plt.scatter(df["ApplicantIncome"], df["Loan_Status"]) #no pattern

#Missing values
df=df.dropna()
#Got a couple of missing values

cat_data = []
num_data = []

for i,c in enumerate(df.dtypes):
    if c == object:
        cat_data.append(df.iloc[:,i])
    else:
        num_data.append(df.iloc[:,i])
cat_data=pd.DataFrame(cat_data).transpose()
num_data=pd.DataFrame(num_data).transpose()


#filling missing numerical values with their median

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan, strategy="median")

imputer.fit(num_data)
num_data=pd.DataFrame(imputer.transform(num_data), columns=num_data.columns)


#Replacing missing values of categorical data by their most frequent occurence

cat_data = cat_data.apply(lambda x:x.fillna(x.value_counts().index[0]))




#No more missing values


#Encoding categorical data with LabelEncoder

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
from sklearn.preprocessing import OneHotEncoder


#target column preprocessing
target_values = {'Y': 1 , 'N' : 0}
target = cat_data['Loan_Status']
cat_data.drop('Loan_Status', axis=1, inplace=True)
target = target.map(target_values)

#other columns
for i in cat_data:
    cat_data[i]=le.fit_transform(cat_data[i])


#Preprocessing done: Concatening the lists

df_preprocessed = pd.concat([cat_data, num_data, target], axis=1)


# Creating Train and Test set

X=df_preprocessed.iloc[:,:-1]
y=df_preprocessed[["Loan_Status"]]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


#Calculating Null Accuracy for comparison and better interpretation
null_acc = y_test.mean()
print(f'Null Accuracy : {null_acc}')

#Part 1: Logistic regression







#logreg.fit(X_train, y_train)

y_pred_class=logreg.predict(X_test)

from sklearn import metrics
acc=metrics.accuracy_score(y_test, y_pred_class)
recall_score = metrics.recall_score(y_test, y_pred_class)
precision = metrics.precision_score(y_test, y_pred_class)
print(f'Accuracy = {acc}')
print(f'Recall Score = {recall_score}')
print(f'Precision = {precision}')


cm = metrics.confusion_matrix(y_test, y_pred_class)


#As we'll be testing different algorithm i'm going to create a function that evaluate models tested
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, accuracy_score
from sklearn.model_selection import cross_val_score


def evaluate_model(y_true, y_pred, retu=False):
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    loss = log_loss(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    if retu:
        return pre, rec, f1, loss, acc
    else:
        print('  pre: %.3f\n  rec: %.3f\n  f1: %.3f\n  loss: %.3f\n  acc: %.3f' % (pre, rec, f1, loss, acc))

#Add models here to test them with our data set
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

models = {
    'LogisticRegression': LogisticRegression(random_state=42),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=1, random_state=42),
    'SVC' : SVC(random_state = 42, probability = True),
    'RandomForest' : RandomForestClassifier(n_estimators = 50)
}
#Evaluate performance on the train set
def train_evaluate(models, X_train, X_test, y_train, y_test):
    for name, model in models.items():
        print(name,':')
        model.fit(X_train, y_train)
        evaluate_model(y_test, model.predict(X_test))
        y_test_prob = model.predict_proba(X_test)[:,1]
        auc= metrics.roc_auc_score(y_test, y_test_prob)
        print(f'\nAUC Score : {auc}')
        print('-'*30)
        
train_evaluate(models, X_train, X_test, y_train, y_test)
"""
 RESULTS: Best model is Logistic Regression, with 
  pre: 0.826
  rec: 0.982
  f1: 0.897
  loss: 5.607
  acc: 0.838
  AUC Score : 0.759899434318039
  
  UPDATE: RandomForestClassifier giving best results (as expected) though hyperaparameters were chosen experimentally.
  Need to GridSearch that.
  
"""





## Grid Searching RandomForestClassifier hyperparameters, validating with cross validation and pipelines!
# Run this piece of code to process with a fresh dataset
import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv(("loan_dataset.csv"))
df=df.dropna()
df=df.drop('Loan_ID',axis=1)

cat_columns = []
num_columns = []
for i,c in enumerate(df.dtypes):
    if c == object:
        cat_columns.append(df.columns[i])
    else :
        num_columns.append(df.columns[i])
       
del(cat_columns[-1])


X=df.iloc[:,:-1]
y=df[["Loan_Status"]]
target_values = {'Y': 1 , 'N' : 0}
y = df[['Loan_Status']]
y['Loan_Status'] = y['Loan_Status'].map(target_values)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.compose import make_column_transformer
column_transf = make_column_transformer(
        (OneHotEncoder(), cat_columns),
        (SimpleImputer(), num_columns)
        )

RandomForest=RandomForestClassifier()
pipe = Pipeline([('pre-processing', column_transf), ('model', RandomForest)])
param_grid = {
        'model__n_estimators' : [5*(i+1) for i in range(20)],
        'model__max_depth' : [i+1 for i in range(15)]}
search = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy')
search.fit(X_train,y_train)
print("Best parameters found on train set:")
print()
print(search.best_params_)

print("Grid scores on development set:")
print()
means = search.cv_results_['mean_test_score']
stds = search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, search.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()
plt.plot(means)
plt.xlabel('n_estimators for RandomForestClassifier')
plt.ylabel('Accuracy')


print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, search.predict(X_test)
print(classification_report(y_true, y_pred))
print()


# Feature Selection

data_corr = pd.concat([X_train, y_train], axis=1)
corr = data_corr.corr()
plt.figure(figsize=(10,7))
sns.heatmap(corr, annot=True);
# Correlation matrix : Strong corellation between Loan_Amount and Applicant Income, and between Credit_History and Loan_Status (=Our best Feature)
# We also have a medium correlation between Married and Dependents 

