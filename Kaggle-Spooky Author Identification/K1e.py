# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 16:45:19 2017

@author: jeeva
"""

import pandas as pd
import numpy as np
import nltk
from nltk import*
import os
import csv

df=pd.read_csv("train.csv")
#creating stop words list


    
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
dt=pd.read_csv("test.csv")

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
op["id"] = dt.ix[:,0]
        
op.to_csv("jeevan5.csv")
        
    

