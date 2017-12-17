# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:52:57 2017

@author: jeeva
"""
import pandas as pd
import math
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

df=pd.read_csv("train.csv")
X=df.ix[:,['id', 'text',]]
Y=df.ix[:,["author"]]

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0,stratify=None)

X_train["author"]=y_train

df=X_train
    
E=['','','']
r=0

A = df["author"].unique()
dim=[]
for i in A:
    dim.append(len(df[df["author"] == i]))
z5=sum(dim)
pa=[i/z5 for i in dim]#probability of that author


for j in range(len(A)):
    p=df["author"]==A[j]
    k= df.ix[p,"text"]
    for i in k:
        E[r]=E[r]+i
    r=r+1

for l in range(len(E)):
    E[l]=E[l].split(" ")
    
    
    

for l in range(len(E)):
    E[l]=[w.lower() for w in E[l]]
    E[l]=[w for w in E[l] if w.isalpha()]
    

d=[{},{},{}]
for l in range(len(E)):
    for w in E[l]:
        if w in d[l]:
            d[l][w]= d[l][w]+1
        else:
            d[l][w]=1

#convering into data frame and filling Nans with zeros
W= pd.DataFrame(d)
W=W.fillna(0)
V=W.shape[1]

sum=W.sum(axis=1)

den=(sum+V)
X=W+1

den=den
for i in X.columns:
    for j in range(3):
        X.ix[j,i]=X.ix[j,i]/den[j]





#test data
X_test["author"]=y_test
dt=X_test

j=list(dt.ix[:,1])
j1=[j[i].lower() for i in range(len(j))]

j2=[j1[i].split(" ") for i in range(len(j1))]

id1=list(dt.ix[:,0])
E1=[]
H1=[]
M1=[]

for i in range(len(j2)):
    E=pa[0]
    H=pa[1]
    M=pa[2]
    for j in range(len(j2[i])):
        if j2[i][j] in X.columns:
            E*=X.ix[0,j2[i][j]]
            H*=X.ix[1,j2[i][j]]
            M*=X.ix[2,j2[i][j]]
        else:
            E*=1/3
            H*=1/3
            M*=1/3
    E1.append(E)
    M1.append(M)
    H1.append(H)

op=pd.DataFrame(np.column_stack([E1,H1,M1]),columns=["EAP","HPL","MWS"])

op1= op.apply(np.sum,axis=1)# finding each row sum

for i in op.columns:
    for j in range(len(op)):
        op.ix[j,i]=op.ix[j,i]/op1[j]


Y_actual=pd.get_dummies(y_test)

n_test=len(Y_actual)

headernames=op.columns

Y_actual.columns = headernames

Y_a=Y_actual.set_index(op.index)


sum1=0
op=op.fillna(op.mean())
for i in range(n_test):
    for j in range(3):
        sum1+=Y_a.ix[i,j]*(math.log(op.ix[i,j]))


error=-sum1/n_test

import seaborn as sns
sns.set()
import matplotlib.pyplot as plt


objects = ('0.9/0.1', '0.8/0.2', '0.7/0.3', '0.6/0.4')
y_pos = np.arange(len(objects))
performance = [0.89, 0.85, 0.80, 0.84]
 
plt.bar(y_pos, performance, align='center', alpha=0.5,color='g')
plt.xticks(y_pos, objects)
plt.ylabel('logloss error')
plt.title('Average Error rate for differnt models')
 
plt.show()



labels = ['0.9/0.1', '0.8/0.2', '0.7/0.3', '0.6/0.4']
sizes = [0.89, 0.85, 0.80, 0.84]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
explode = (0, 0, 0.1, 0)
patches, texts = plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()

