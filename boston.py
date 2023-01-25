# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 08:46:29 2023

@author: 21pd08
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
 



df=pd.read_csv("boston.csv")
pd.set_option('display.max_columns', None)
print(df.describe())
corre=df.corr()
print(corre.shape)
print(corre)
sns.heatmap(corre,vmin=-1,vmax=1,cmap='bwr_r') #RdPu



#plot between rrom and price
plt.plot(df['MEDV'],df['RM'])
plt.xlabel('Price',col='black')
plt.ylabel


"""
y=df['MEDV']
x=df.drop(['MEDV'],axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=0)
#print(x_train,x_test)
reg=LinearRegression()
reg.fit(x_train,y_train)
print(reg.coef_)
print(reg.intercept_)
predict=reg.predict(x_test)
plt.plot(predict,label="Prediction")
plt.plot(y_test.to_numpy(),label="Actual")
plt.legend()
plt.title("Actual vs Predicted")
#print(df.head(5))
#print(df.dtypes)

"""

"""

print(df.isna().sum())
age=df['AGE']
plt.plot(age)
"""