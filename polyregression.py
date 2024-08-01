import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures 


# polynomial regresssion

# It is the of equation it is used find the relationship between the depndendt 
# and independent variable it can be both linear and non-linear

# Linear polynomial regression
 
x=np.arange(-5.0,5.0,0.2)

y= 2 *(x) + 3
 
#  cause here the function is linear function

y_noise=2 * np.random.normal(size=x.size)

y_data=y+ y_noise

plt.scatter(x,y_data,color='blue')
plt.plot(x,y,color='red')
plt.show()


# NonLinear polynomial regression

x1=np.arange(-5.0,5.0,0.2)

y1=2*(x1 **2) +1*(x1**1) + 6

y_noise1=2 * np.random.normal(size=x1.size)

y_data1=y1 + y_noise1

plt.scatter(x1,y_data1,color='blue')
plt.plot(x1,y1,color='red')
plt.show()

data=pd.read_csv('data1.csv')
# print(data.head())

data_col=data[['runs','matches']]
# data_col.hist()
# plt.show()

target_col=data['centureis']
# print(target_col)

x_train,x_test,y_train,y_test=train_test_split(data_col,target_col,test_size=0.2,random_state=4)
# print(x_train)

linemodel=linear_model.LinearRegression()
linemodel.fit(x_train,y_train)

final_predection=linemodel.predict(x_test)

print('Mean squared Error',np.mean((final_predection - y_test) **2))
print('Mean absolute Error',np.mean(np.absolute(final_predection  - y_test)))
print('r2 scroe',r2_score(y_test,final_predection))

runs=1000
matches=14

sample_data=np.array([[runs,matches]])
# print(sample_data)  
sample_pred=linemodel.predict(sample_data)

print('The predicted centureis is ',sample_pred)

print('****************')

#Using Linear regression to make predection

pol_train=PolynomialFeatures(degree=2)
x_train_poly=pol_train.fit_transform(x_train)
x_test_poly=pol_train.fit_transform(x_test)

# Polynomial model training and fiting

polymodel=linear_model.LinearRegression()
polymodel.fit(x_train_poly,y_train)

polypredection=polymodel.predict(x_test_poly)

print('Mean squared Error',np.mean((polypredection - y_test) **2))
print('Mean absolute Error',np.mean(np.absolute(polypredection  - y_test)))
print('r2 scroe',r2_score(y_test,polypredection))

runs1=1000
matches1=14

sample_data1=np.array([[runs1,matches1]])
sample_data_poly=pol_train.fit_transform(sample_data1)
final_poly_prediction= polymodel.predict(sample_data_poly)

print('The predicted centuries by polynomial regression is ',final_poly_prediction)