import pandas as pd
import numpy as np
import nltk
from nltk import*

import pip
pip.main(['install','textmining','re','stemmer','csv','os'])

import textmining

"""import pip
stop words list
http://xpo6.com/download-stop-word-list/      
pip.main(['install','nltk'])

from nltk.corpus import stopwords
s=stopwords.words("english")"""

df=pd.read_csv("train.csv")
#creating stop words list
s=pd.read_csv("stop-word-list.csv",header=None)
stop=[]
for i in range(119):
    stop.append(s.ix[0,i])





    
E=['','','']
r=0

A = df["author"].unique()

for j in range(len(A)):
    p=df["author"]==A[j]
    k= df.ix[p,"text"]
    for i in k:
        E[r]=E[r]+i
    r=r+1

tdm = textmining.TermDocumentMatrix()
