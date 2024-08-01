import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures,LabelEncoder,StandardScaler
from sklearn.model_selection import  train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


data=pd.read_csv('housedata.csv')
# print(data.head())

indep_columns=data[['bedrooms','sqft_basement','sqft_lot']].astype(float)

label_encoder=LabelEncoder()
scaler=StandardScaler()

# here this method is used for normalize and standardize the data


# indep_columns['sqft_basement']=label_encoder.fit_transform(data['sqft_basement'].astype(float))
# indep_columns['sqft_lot']=label_encoder.fit_transform(data['sqft_lot'].astype(float))
indep_columns['city_num']=label_encoder.fit_transform(data['city'])
indep_columns['city_num']=indep_columns['city_num'].astype(float)
indep_columns.hist()
# plt.show()

print(indep_columns)


target_columnn=data['price']

x_train,x_test,y_train,y_test=train_test_split(indep_columns,target_columnn,test_size=0.2,random_state=3)

x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

model=linear_model.LinearRegression()
model.fit(x_train,y_train)

predect1=model.predict(x_test)



print('Mean squared Error',mean_squared_error(y_test,predect1))
print('Mean absolute Error',mean_absolute_error(y_test,predect1))
print('R2 score',r2_score(y_test,predect1))



# PolynomialFeatures

pol_transform=PolynomialFeatures(degree=2)
x_poly_train=pol_transform.fit_transform(x_train)
x_poly_test=pol_transform.fit_transform(x_test)

poly_model=linear_model.LinearRegression()
poly_model.fit(x_poly_train,y_train)

final_prediction=poly_model.predict(x_poly_test)

print('Mean squared Error',mean_squared_error(y_test,final_prediction))
print('Mean absolute Error',mean_absolute_error(y_test,final_prediction))
print('R2 score',r2_score(y_test,final_prediction))

sqft_lot=7712
bedrooms=3.0
sqft_basement=360
city='Seattle'
sample_numeric=label_encoder.fit_transform([city])[0]
print(sample_numeric)

sampledata=np.array([[sqft_lot,bedrooms,sqft_basement,sample_numeric]])

sampledata=scaler.transform(sampledata)

sampledata_poly=pol_transform.fit_transform(sampledata)

sampleprediction= poly_model.predict(sampledata_poly)

samplelinearmodel=model.predict(sampledata)

print('Predicted price is ',sampleprediction[0])

print('Predicted price by linear regression is',samplelinearmodel[0])

# Mean squared Error 845130547108.0762
# Mean absolute Error 243312.15358910424
# R2 score 0.012991177570423007
# Mean squared Error 844062989682.2329
# Mean absolute Error 236151.43316650495
# R2 score 0.014237953706208395 
# accuracy of both regression methods without normalization