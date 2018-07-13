import numpy as np
import pandas as pd
from sklearn import linear_model,cross_validation,metrics 
import matplotlib.pyplot as plt
data=pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv',index_col=0)
#plt.scatter(data['TV'],data['sales'],color='b')
#plt.ylabel('sales')
#plt.show()
#plt.scatter(data['newspaper'],data['sales'],color='b')
#plt.xlabel('newpaper')
#plt.ylabel('sales')
#plt.show()
#plt.scatter(data['radio'],data['sales'],color='b')
#plt.xlabel('radio')
#plt.ylabel('sales')
#plt.show()

y=data['sales']
"""
x=data[['TV','radio','newspaper']]
x_train,x_test,y_train,y_test=cross_validation.train_test_split(x,y,test_size=0.2)

clf=linear_model.LinearRegression()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print(clf.intercept_)
print(clf.coef_)
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

"""
# considering tv and radio as features

x=data[['TV','radio']]
x_train,x_test,y_train,y_test=cross_validation.train_test_split(x,y,random_state=1)

clf=linear_model.LinearRegression()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print(clf.intercept_)
print(clf.coef_)
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))







