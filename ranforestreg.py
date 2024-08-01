import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

x,y=make_regression(n_samples=100,n_features=2,noise=0.1,random_state=42)
# print(x)
# print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

model=RandomForestRegressor(n_estimators=100,random_state=42)

model.fit(x_train,y_train)
predection=model.predict(x_test)


print('Mean squared error',mean_squared_error(y_test,predection))
# y_train,y_test=train_test_split