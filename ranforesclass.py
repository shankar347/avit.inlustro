from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
import matplotlib.pyplot as plt


iris=load_iris()
x=iris.data
y=iris.target

# print(x)
# print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

model=RandomForestClassifier(n_estimators=100,random_state=42)

model.fit(x_train,y_train)
predection=model.predict(x_test)

plt.plot(x,color="blue")
plt.plot(predection,color='red')
plt.show()

print('accuracy score',accuracy_score(y_test,predection))
print('classification report',classification_report(y_test,predection))


import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Generate a synthetic regression dataset
X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)