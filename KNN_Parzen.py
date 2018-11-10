# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 21:27:36 2018

@author: Alexandra
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing as pr
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import pyplot as plt

def get_grid(data):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))


df = pd.read_csv('C:/Users/Alexandra/Anaconda3/Scripts/machine learning/adult_data.txt', sep=',')
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation','relationship', 'race', 'sex', 'capital-gain','capital-loss', 'hours-per-week', 'native-country', 'result']
df1 = df.copy()
df1 = df1.drop('result', 1)
numberic_columns = df1.columns[df1.dtypes != 'object']
categorical_columns = df1.columns[df1.dtypes == 'object']

scaler = pr.StandardScaler()
numberic_df = pd.DataFrame(scaler.fit_transform(df1[numberic_columns]))

label_encoder = pr.LabelEncoder()
for column in categorical_columns:
    df1[column] = label_encoder.fit_transform(df1[column])

onehot_encoder = pr.OneHotEncoder(sparse=False)
categorical_df = pd.DataFrame(onehot_encoder.fit_transform(df1[categorical_columns]))
#print(categorical_df.head())

encoded_df = pd.concat([categorical_df, numberic_df], axis=1)

#print(encoded_df.head())

Y = label_encoder.fit_transform(df['result'])
X_train, X_holdout, y_train, y_holdout = train_test_split(df1.values, Y, test_size=0.3, random_state = 17)
knn = KNeighborsClassifier(n_neighbors=30, weights='distance')
#knn1 = RadiusNeighborsClassifier(LeaveOneOut().get_n_splits(X_train)).fit(X_train, y_train)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_holdout)
#knn1_pred = knn1.predict(X_holdout)
print(accuracy_score(y_holdout, knn_pred))
#print(accuracy_score(y_holdout, knn1_pred)) 
#knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=-1))])
#knn_params = {'knn__n_neighbors': range(1, ???)}
#knn_grid = GridSearchCV(knn_pipe, knn_params, cv = 5, n_jobs= -1, verbose=True)
#knn_grid.fit(X_train, y_train)
#knn_grid.best_params_, knn_grid.best_score_
#accuracy_score(y_holdout, knn_grid.predict(X_holdout))
