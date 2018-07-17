import numpy as np
import pandas as pd

data = pd.read_csv("dataset.csv")
x=data['l1']
y=data['l2']
X=np.array(list(zip(x,y)))
X=np.transpose(X)
sigma=np.cov(X)
u,s,vh=np.linalg.svd(sigma)
U_reduce = u[:,:1]
u_reduce=np.transpose(U_reduce)
z=np.matmul(u_reduce,X)
print(np.transpose(z))
