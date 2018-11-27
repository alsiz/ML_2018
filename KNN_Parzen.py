# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 21:27:36 2018

@author: Alexandra
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
%matplotlib inline

adults = pd.read_csv('adult_data.txt', sep = ',', names=['Age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','label'])
adults_test = pd.read_csv('adult_data.txt', sep = ',', names=['Age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','label'])
adults_test_1 = pd.read_csv('adult_data.txt', sep = ',', names=['Age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','label'])
adults_test_1.columns = adults_test_1.columns.str.replace('.','_')
df_obj = adults_test_1.select_dtypes(['object'])
adults_test_1[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

train_data = adults.drop('label',axis=1)
test_data = adults_test.drop('label',axis=1)
data = train_data.append(test_data)
label = adults['label'].append(adults_test['label'])

full_dataset = adults.append(adults_test)
data_binary = pd.get_dummies(data)


x_train, x_test, y_train, y_test = train_test_split(data_binary,label)

performance = []
knn_scores = []
train_scores = []
test_scores = []

for n in range(1,20,2):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(x_train,y_train)
    train_score = knn.score(x_train,y_train)
    test_score = knn.score(x_test,y_test)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print(f'KNN : Training score - {train_score} -- Test score - {test_score}')
    knn_scores.append({'algorithm':'KNN', 'training_score':train_score})
    
plt.scatter(x=range(1, 20, 2),y=train_scores,c='b')
plt.scatter(x=range(1, 20, 2),y=test_scores,c='r')

plt.show()
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)

knn.score(x_train,y_train)

train_score = knn.score(x_train,y_train)
test_score = knn.score(x_test,y_test)

print(f'K Neighbors : Training score - {train_score} - Test score - {test_score}')

performance.append({'algorithm':'K Neighbors', 'training_score':train_score, 'testing_score':test_score})

adults_test_1['sex_map'] = adults_test_1.sex.map({'Female':0, 'Male':1})
adults_test_1['race_map']=adults_test_1.race.map({'White':0, 'Black':1, 
                                    'Asian-Pac-Islander':2, 'Amer-Indian-Eskimo':3})
adults_test_1['marital_map'] = adults_test_1.marital_status.map({'Widowed':0, 'Divorced':1, 'Separated':2,
                                                  'Never-married':3, 'Married-civ-spouse':4,
                                                  'Married-spouse-absent':5, 'Married-AF-spouse':6})
adults_test_1['rel_map']=adults_test_1.relationship.map({'Not-in-family':0, 'Unmarried':0, 
                                           'Own-child':0, 'Other-relative':0, 
                                           'Husband':1, 'Wife':1})
adults_test_1['work_map']=adults_test_1.workclass.map({'?':0, 'Private':1, 'State-gov':2, 'Federal-gov':3, 
                                        'Self-emp-not-inc':4, 'Self-emp-inc': 5, 'Local-gov': 6,
                                        'Without-pay':7, 'Never-worked':8})
adults_test_1['label_map']=adults_test_1.label.map({'<=50K':0, '>50K': 1})
num_cols = ['Age', 'sex_map', 'race_map','education_num', 'work_map', 
            'marital_map', 'rel_map', 'hours_per_week','capital_gain', 'capital_loss', 
            'fnlwgt', 'label_map']
adults_test_1=adults_test_1[num_cols].fillna(-9999)
features=['Age', 'sex_map', 'race_map','education_num', 'work_map', 
          'marital_map', 'rel_map', 'hours_per_week','capital_gain', 
          'capital_loss', 'fnlwgt']
target=['label_map']
X = adults_test_1[features].values
y = adults_test_1[target].values.flatten()
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X,y, random_state=0)

def logistic_regression():
    
    # Instantiates and trains a logistic regression model using grid search  
    # to find optimalhyperparameter values 
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    
    logreg = LogisticRegression()
    grid_values = {'penalty' : ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100]}
    
    grid_lr_rec = GridSearchCV(logreg, param_grid = grid_values, scoring = 'accuracy')
    grid_lr_rec.fit(X, y)
    
    return grid_lr_rec.best_estimator_
def roc_curve(model):
    from sklearn.metrics import roc_curve, auc
    
    #scores = model.decision_function(X_test)
    proba = model.predict_proba(X_test_1)
    
    fpr, tpr, _ = roc_curve(y_test_1, proba[:,1])
    
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve. AUC: {}'.format(auc(fpr, tpr)))
  roc_curve(logreg)
logreg = logistic_regression()
