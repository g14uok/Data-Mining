# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 21:20:14 2017

@author: jeeva
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 20:23:32 2017

@author: jeeva
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 12:57:01 2017

@author: jeeva
"""

import pandas as pd 
import json
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import svm

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
X=X1

#X=np.array(X1)

   
y=df.ix[:,"is_iceberg"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
"""svm model
clf = svm.SVC()
clf.fit(X_train, y_train)

y_predict=clf.predict(X_test)

"""
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
neigh = KNeighborsClassifier(n_neighbors=3)
from sklearn.model_selection import cross_val_score
neigh.fit(X_train, y_train) 
y_predict=neigh.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_predict, labels=[0,1]).ravel()
print((tn, fp, fn, tp))
accuracy=100*(tp+fn)/len(y_predict)
print(accuracy)
sns.set()


scores = cross_val_score(neigh, X, y, cv=10)
print(100*np.mean(scores))



objects = ("Accuracy without scaling the data","Accuracy after scaling the data")
y_pos = np.arange(len(objects))
performance = [53,89.29]
 
plt.bar(y_pos, performance, align='center', alpha=0.5,color='r')
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Deep Neural Network model accuracy')
 
plt.show()



objects = ("3","4","5","6","7","8","9","10")
y_pos = np.arange(len(objects))
performance = [0.7842,0.7867,0.7873,0.79,0.789,0.7898,0.7923,0.7929]
 
plt.bar(y_pos, performance, align='center', alpha=0.5,color='g')
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Performing Vfold cross validation')
 
plt.show()