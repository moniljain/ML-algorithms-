import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,cross_validation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
def make_meshgrid(x,y):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max,0.02))
    return xx,yy
data=pd.read_csv("data_classification.csv")
X=data[['Study','sleep']]
y=data['result']
x_train,x_test,y_train,y_test=cross_validation.train_test_split(X,y,train_size=0.8)
para_tuning={'kernel':['linear'],'C':[10,11,12],'gamma':[0.1,0.2,0.5,0.7,0.9,1]}
clf=svm.SVC()
g=GridSearchCV(clf,para_tuning)
g.fit(x_train,y_train)
y_pred=g.predict(x_test)
print(accuracy_score(y_test,y_pred))
xx=data['Study']
yy=data['sleep']
x1=[]
y1=[]
x2=[]
y2=[]
for i in range(len(y)):
    if y[i]==1:
        x1.append(xx[i])
        y1.append(yy[i])
    else:
        x2.append(xx[i])
        y2.append(yy[i])
plt.scatter(x1,y1,c='green')
plt.scatter(x2,y2,c='red')
plt.show()



