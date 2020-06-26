# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 09:48:41 2020

@author: Pierre Cugnet

Project Description: Personnal project - The aim of this project if to test out different basic algorithm, and to implemente pipeline and hyperparameters tuning for process validation with the best classifier.
Basic pre-processing is done, though this is not the objective.
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
sns.countplot(x="Self_Employed", hue="Loan_Status", data=df)
"""
#plt.scatter(df["ApplicantIncome"], df["Loan_Status"]) #no pattern

# Feature Exploration





## Target value pre-procssing : Transforming Y and N by 1 and 0
df['Loan_Status'] = df['Loan_Status'].apply(lambda x : x.replace('Y', '1').replace('N', '0')).astype(int)


#Missing values
df.isna().mean()

#Got a couple of missing values not that much though (8% for Credit_History is the max): I'm going to consider that missing values are relevant informations

#Splitting categorical and numerical features into 2 different datasets
cat_data = pd.DataFrame()
num_data = pd.DataFrame()    
cat_data=df.select_dtypes(include = 'object')
num_data=df.select_dtypes(include = 'number')
target=num_data.iloc[:,-1]
num_data.drop("Loan_Status", axis=1, inplace=True)


#filling missing numerical values with their median and scaling them

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan, strategy="median")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

imputer.fit(num_data)
num_data = pd.DataFrame(imputer.transform(num_data), columns=num_data.columns)

scaler.fit(num_data)
num_data = pd.DataFrame(scaler.transform(num_data), columns=num_data.columns)


#Encoding categorical data with dummies (to keep nan as relevant features)
cat_data=pd.get_dummies(cat_data, drop_first=True)


#Preprocessing done: Concatening the lists

df_preprocessed = pd.concat([cat_data, num_data, target], axis=1)
corr = df_preprocessed.corr()
plt.figure(figsize=(10,7))
sns.heatmap(corr, annot=True)
# Correlation matrix : Strong corellation between Loan_Amount and Applicant Income, and between Credit_History and Loan_Status (=Our best Feature)
# This dataset is very imbalanced and the only relevant feature is Credit_History, we could actually drop all the features except that one but that's not the purpose of this project



# Creating Train and Test set

X=df_preprocessed.iloc[:,:-1]
y=df_preprocessed[["Loan_Status"]]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


#Calculating Null Accuracy for comparison and better interpretation
null_acc = y_test.mean()
print(f'Null Accuracy : {null_acc}')

#Part 1: Logistic regression example






from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)

y_pred_class=logreg.predict(X_test)

from sklearn import metrics
acc=metrics.accuracy_score(y_test, y_pred_class)
recall_score = metrics.recall_score(y_test, y_pred_class)
precision = metrics.precision_score(y_test, y_pred_class)
print(f'Accuracy = {acc}')
print(f'Recall Score = {recall_score}')
print(f'Precision = {precision}')


cm = metrics.confusion_matrix(y_test, y_pred_class)
print(cm)

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
        print('pre: %.3f\n  rec: %.3f\n  f1: %.3f\n  loss: %.3f\n  acc: %.3f' % (pre, rec, f1, loss, acc))

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
df=df.drop('Loan_ID',axis=1)

cat_columns = []
num_columns = []
for i,c in enumerate(df.dtypes):
    if c == object:
        cat_columns.append(df.columns[i])
    else :
        num_columns.append(df.columns[i])
       
del(cat_columns[-1]) #Del target column


X=df.iloc[:,:-1]
y = df['Loan_Status']
y= y.apply(lambda x : x.replace('Y', '1').replace('N', '0')).astype(int) #Replacing Y and N by 1 and 0


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix

# Using a Cross validation process, we need to be careful of data leakage ! pipeline help us doing that!
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_columns),
        ('cat', categorical_transformer, cat_columns)])

RandomForest=RandomForestClassifier()

pipe = Pipeline([('pre-processing', preprocessor), ('model', RandomForest)])
param_grid = {
        'model__n_estimators' : [10,25,50,100],
        'model__max_depth' : [i+1 for i in range(15)],
        }
search = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy', refit=True)
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


print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, search.predict(X_test)
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true,y_pred))
print()



