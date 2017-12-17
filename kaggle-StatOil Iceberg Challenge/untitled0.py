# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:50:26 2017

@author: jeeva
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.model_selection import train_test_split



df=pd.read_excel("input.xlsx")
df=pd.get_dummies(df)
df=df.astype(float)

Y=df["cost_per_click"]
X=df.drop("cost_per_click",axis=1)
X=preprocessing.scale(X)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
model = Sequential()
model.add(Dense(68, input_dim=68, activation='sigmoid'))
model.add(Dense(8, activation='tanh'))
model.add(Dense(1, activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam')


model.compile(loss='mean_absolute_error', optimizer='rmsprop')

model.fit(X_train, y_train, nb_epoch=20, batch_size=100)
score = model.evaluate(X_test, y_test, batch_size=100)

