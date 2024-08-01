import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,mean_squared_error,r2_score

data = {
    'radius_mean': [17.99, 20.57, 19.69, 11.42, 20.29],
    'texture_mean': [10.38, 17.77, 21.25, 20.38, 14.34],
    'perimeter_mean': [122.80, 132.90, 130.00, 77.58, 135.10],
    'area_mean': [1001.0, 1326.0, 1203.0, 386.1, 1297.0],
    'smoothness_mean': [0.11840, 0.08474, 0.10960, 0.14250, 0.10030],
    'compactness_mean': [0.27760, 0.07864, 0.15990, 0.28390, 0.13280],
    'concavity_mean': [0.30010, 0.08690, 0.19740, 0.24140, 0.19800],
    'concave_points_mean': [0.14710, 0.07017, 0.12790, 0.10520, 0.10430],
    'symmetry_mean': [0.24190, 0.18120, 0.20640, 0.25970, 0.18090],
    'fractal_dimension_mean': [0.07871, 0.05667, 0.05999, 0.09744, 0.05883],
    'diagnosis': ['M', 'M', 'M', 'B', 'M']
}
# print(data)
df=pd.DataFrame(data)
# print(df.head())

feature_col=df.drop('diagnosis',axis=1)
# print(feature_col)

target_col=df['diagnosis'].map({'M':1,'B':0})
# print(target_col)

x_train,x_test,y_train,y_test=train_test_split(feature_col,target_col,test_size=0.3,random_state=42)
log_model=LogisticRegression()

log_model.fit(x_train,y_train)

predection=log_model.predict(x_test)

print('Mean squared error',mean_squared_error(y_test,predection))
print('R2score',r2_score(y_test,predection))
print('Classification report',classification_report(y_test,predection))


