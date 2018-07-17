import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from copy import deepcopy
def e_dist(x1,y1,x2,y2):
    return sqrt((x1-x2)**2 + (y1-y2)**2)
    
data = pd.read_csv("dataset.csv")
x=data['l1']
y=data['l2']
X = np.array(list(zip(x,y)))
plt.scatter(x,y,c='black',s=7)
plt.show()
k=2
clusters = np.zeros(len(X))
c_x=x[np.random.randint(0,len(x)-1,size=k)]
c_y=y[np.array(np.random.randint(0,len(y)-1,size=k))]
c=np.array(list(zip(c_x,c_y)),dtype=np.float32)
c_old=np.zeros(c.shape)
plt.scatter(x,y,c='black',s=7)
plt.scatter(c_x,c_y,marker='*',c=['g','r'],s=200)
plt.show()
#print(c)
dist = np.zeros(len(c))
error = np.linalg.norm(c-c_old,axis=None)
#print(error)
while error!=0:
    for i in range(len(X)):
        dist=np.linalg.norm(X[i]-c,axis=1) 
        clusters[i]=np.argmin(dist)
    c_old = deepcopy(c)
    for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        c[i]=np.mean(points,axis=0)
    error = np.linalg.norm(c-c_old,axis=None)

colors = ['r','g']
for i in range(k):
    points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
    plt.scatter(points[:,0],points[:,1],c=colors[i],s=7)
plt.scatter(c[:,0],c[:,1],s=200,c='black',marker='*')
plt.show()
    
