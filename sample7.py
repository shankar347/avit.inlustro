import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


data=pd.read_csv('data1.csv')
# print(data)
x=data[['runs','wickets','centureis']]
# print(x)
y=data['matches']
# print(y)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=60)

model=LinearRegression()
model.fit(x_train,y_train)


y_pred=model.predict(x_test)

print(y_pred)
ms=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print(f'Mean squared Error :{ms}')
print(f'R-squared:{r2}')



