import pandas as pd
import numpy as np
from sklearn import cross_validation,preprocessing,neighbors,datasets
iris=datasets.load_iris()
X=iris.data
y=iris.target
X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.33)
kk=[]
for i in range(1,50,2):
    kk.append(i)
cv_scores=[]
for k in kk:
    knn=neighbors.KNeighborsClassifier(n_neighbors=k)
    score=cross_validation.cross_val_score(knn,X_train,y_train,cv=10,scoring='accuracy')
    cv_scores.append(score.mean())
op_k=kk[cv_scores.index(max(cv_scores))];
print(op_k)
clf=neighbors.KNeighborsClassifier(n_neighbors=op_k)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(y_pred)
print(y_test)
accuracy=clf.score(X_test,y_test)
print(accuracy)
 

