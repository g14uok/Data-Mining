
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 12:57:01 2017

@author: jeeva
"""
from sklearn import preprocessing
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
"""df=df.drop(["id","band_2"],axis=1)"""
df.info()
df=df.replace("na",40)


dataa=[]
for i in range(1604):
    data1=df.ix[i,0]
    dataf=np.reshape(data1,(75,75))
    dataf2=dataf[20:50,20:50]
    dataf3=np.reshape(dataf2,(900,))
    dataa.append(dataf3)

dataa=np.array(dataa)   
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
datab=[]
for i in range(1604):
    data1=df.ix[i,1]
    dataf=np.reshape(data1,(75,75))
    dataf2=dataf[20:50,20:50]
    dataf3=np.reshape(dataf2,(900,))
    datab.append(dataf3)

datab=np.array(datab) 



df1=pd.DataFrame(dataa)
df2=pd.DataFrame(datab)
df2.columns=np.arange(900,1800)
X1=pd.concat([df1, df2], axis=1)
X1[1800]=df["inc_angle"]
#X=pd.concat([X1, df["inc_angle"]], axis=1)


X=np.array(X1)
X=preprocessing.scale(X)
   
y=df.ix[:,"is_iceberg"]


    


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# create model
model = Sequential()
model.add(Dense(1801, input_dim=1801, activation="linear"))
model.add(Dense(700, activation='softplus'))

model.add(Dense(100, activation='tanh'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=30, batch_size=100)



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
    
    


