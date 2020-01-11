import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data=pd.read_csv('train.csv')
x=data.iloc[:,1:]
y=data.iloc[:,0]

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=5)
# model = svm.LinearSVC(random_state=20)

model.fit(x_train,y_train)
predicted= model.predict(x_test)

score=accuracy_score(y_test,predicted)
print("Your Model Accuracy is", score)
