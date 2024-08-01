import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score,classification_report,accuracy_score

raw_data={
"email":["Congrulation! you got new offter ",
         "New Free reward for you",
       "Where are you siva",
       "Free! apply you bank details",
       "Welcome to college",
       "Play winzo win cass price",
    #      "come to college for play cricket",
    #    "Provide your bank details to get free money",
    #    "Provide your aadar details for college",
    #    "Click this link to get free points",
    #    "Upload your bank details in this form"
     ],
"label":[1,1,0,1,0,1]
}

data=pd.DataFrame(raw_data)
print(data.head())

x=data['email']
y=data['label']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

vectorizer=TfidfVectorizer()

x_vect_train=vectorizer.fit_transform(x_train)
x_vect_test=vectorizer.transform(x_test)

model=LogisticRegression()

model.fit(x_vect_train,y_train)

predection=model.predict(x_vect_test)

print('Accuracy',accuracy_score(y_test,predection))
print('Classification report',classification_report(y_test,predection))

sample_data=["Congulation! you got free access","Free!! reward","Welcome to Meeting"]

single=["Free!! reward",
        "Hello Bob",
        "Your purchase was success",
        "Play rummy jungle",
        "Free!! get more money",
        "Won case price ",
        "pruchase was failed"
        # ,"Come to groud for play football",
        # "Win cass price "
        ]
single_vect=vectorizer.transform(single)
sample_predection=model.predict(single_vect)

for predection,email in zip(sample_predection,single):
 if (predection == 0):
  print(f"{email} is Not a scam email")
 else:
  print(f"{email} is  Scame email") 

#   its perfectly working


