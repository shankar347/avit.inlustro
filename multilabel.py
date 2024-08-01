import pandas as pd
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import MultiOutputClassifier

data = {
    'Title': ['Movie1', 'Movie2', 'Movie3', 'Movie4'],
    'Description': ['Action packed thriller',
                    'A romantic comedy',
                    'Horror movie with suspense',
                    'Action and drama combined'],
    'Action': [1, 0, 0, 1],
    'Comedy': [0, 1, 0, 0],
    'Drama': [0, 0, 0, 1],
    'Horror': [0, 0, 1, 0],
    'Romance': [0, 1, 0, 0]
}

df=pd.DataFrame(data)
# print(df.head())

x=df['Description']
y=df[['Action','Comedy', 'Drama', 'Horror', 'Romance']]



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

vectorizer=CountVectorizer()

x_vectorized_train=vectorizer.fit_transform(x_train)
x_vectorized_test=vectorizer.transform(x_test)

model=MultiOutputClassifier(LogisticRegression())

model.fit(x_vectorized_train,y_train)

predection=model.predict(x_vectorized_test)

print('Accuracy',accuracy_score(y_test,predection))
print('Classification report',classification_report(y_test,predection,target_names=y.columns))

# print('Cross validation',cross_val_score(model,x_vectorized,y))