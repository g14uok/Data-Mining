# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 12:57:01 2017

@author: jeeva
"""

import pandas as pd 
import json
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)



with open("train.json") as f:
    d=json.loads(f.read())

df=pd.DataFrame(d)
df=df.drop(["id","band_2"],axis=1)
df.info()
data=[]
for i in range(1604):
    data1=df.ix[i,0]
    dataf=np.reshape(data1,(75,75))
    dataf2=dataf[20:50,20:50]
    dataf3=np.reshape(dataf2,(900,))
    data.append(dataf3)

data=np.array(data)   
"""    
names=[i for i in range(1604)]

X=pd.DataFrame()
for i in range(1604):
    data1=df.ix[i,0]
    dataf=np.reshape(data1,(75,75))
    FD=dataf[20:50,20:50].reshape(900,1)
    for j in range(900):
        X.ix[i,j]=FD[j]
    

data=[]
for i in range(1604):

    data.append(df.ix[i,0])
    
"""


    
y=df.ix[:,"is_iceberg"]


X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.33, random_state=42)

# create model
model = Sequential()
model.add(Dense(1500, input_dim=900, activation="linear"))
model.add(Dense(750, activation='tanh'))
model.add(Dense(300, activation='selu'))
model.add(Dense(100, activation='elu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=20, batch_size=100)



# evaluate the model
scores = model.evaluate(X_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))













"""
df.describe()
df.isnull().sum()
data1=df.ix[5,0]
dataf=np.reshape(data1,(75,75))
data=dataf[20:50,20:50]



plt.imshow(data, interpolation='nearest')
plt.show()



plt.figure(figsize=(2, 10))
for i in range(10):
    data1=df.ix[i,0]
    data=np.reshape(data1,(75,75))
    plt.subplot(2,10,i+1)
    plt.imshow(data, interpolation='nearest')
    
    data1=df.ix[i,1]
    data=np.reshape(data1,(75,75))
    plt.subplot(2,10,i+1+10)
    plt.imshow(data, interpolation='nearest')

plt.show()


"""

   

"""
data1=df.ix[0,0]
dataf=np.reshape(data1,(75,75))
dataf2=dataf[20:50,20:50]
dataf3=np.reshape(dataf2,(900,))
data.append(dataf3) 
 """   
    
    


