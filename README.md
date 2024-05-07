# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import chardet
2. Read the dataset
3. Import SVC from sklearn
4. Fit the data in the model and run the algorithm

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SINDHUJA P
RegisterNumber: 212222220047 
*/

import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding='Windows - 1252')

data.head()

data.info()

data.isnull().sum()

x=data['v1'].values
y=data['v2'].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

## Result Output

![280462441-7fc28e96-c69c-425f-aa26-bbd633dd44c0](https://github.com/Sindhuja9585/Implementation-of-SVM-For-Spam-Mail-Detection/assets/122860624/3e1ece3b-50ca-4daa-ad3d-afbb0cee590f)

## DATA.HEAD()

![280462468-747c9703-064a-4fbd-9f09-c685c2339df9](https://github.com/Sindhuja9585/Implementation-of-SVM-For-Spam-Mail-Detection/assets/122860624/5464cbd8-4082-4768-b31c-c05acd8d81cc)

## DATA.INFO()

![280462528-09533dd4-7f5e-4493-9cba-35091a46d7d1](https://github.com/Sindhuja9585/Implementation-of-SVM-For-Spam-Mail-Detection/assets/122860624/ee0237c0-b119-4141-9f29-4c0e2a3dbc92)

## DATA.ISNULL().SUM()

![280462584-04619be3-255a-40cf-a6a7-5009c4a7820e](https://github.com/Sindhuja9585/Implementation-of-SVM-For-Spam-Mail-Detection/assets/122860624/735f0b01-e71d-473e-bbff-b808a0161507)

## Y_PREDICTION VAUE

![280462642-0bbcc78e-5645-4e4f-91f4-e584ea372e7f](https://github.com/Sindhuja9585/Implementation-of-SVM-For-Spam-Mail-Detection/assets/122860624/f6acfc9a-4513-4ace-9638-23ef3e8f2042)

## ACCURACY VALUE

![280462653-bf4518d1-2af2-4b8e-a582-bbd3b4178d41](https://github.com/Sindhuja9585/Implementation-of-SVM-For-Spam-Mail-Detection/assets/122860624/bbddf713-3a45-4f83-925e-0e17f5610b03)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
