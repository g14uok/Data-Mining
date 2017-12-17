import pandas as pd
import numpy as np


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
            
# removing the stopping words and getting only the main words         
F=[{},{},{}]
for l in range(len(d)):
    for i in d[l]:
        if i not in stop:
            F[l][i]=d[l][i]
            

#convering into data frame and filling Nans with zeros
A= pd.DataFrame(F)
A=A.fillna(0)

A1= A.apply(np.sum,axis=0)# finding each column sum
A=A/A1#finding proportion in each row






    



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
    E=0
    H=0
    M=0
    for j in range(len(j2[i])):
        if j2[i][j] in A.columns:
            E+=A.ix[0,j2[i][j]]
            H+=A.ix[1,j2[i][j]]
            M+=A.ix[2,j2[i][j]]
        else:
            E+=1/3
            H+=1/3
            M+=1/3
    E1.append(E/len(j2[i]))
    M1.append(M/len(j2[i]))
    H1.append(H/len(j2[i]))

op=pd.DataFrame(np.column_stack([dt.ix[:,0],E1,H1,M1],columns=["id","EAP","HPL","MWS"])
#op.to_csv("jeevan.csv")
            







    

    



