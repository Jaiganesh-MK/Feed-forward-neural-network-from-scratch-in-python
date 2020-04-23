import ffnn 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split

df = pd.read_csv("/home/mk/Desktop/feed_forward_neural_net/titanic.csv",header = None)
y = df[0]
x = df.drop([0],axis = 1)
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 22)
X = X_train.to_numpy().T
y = y_train.to_numpy().reshape(-1,1).T
X_test = X_test.to_numpy().T
y_test = y_test.to_numpy().reshape(-1,1).T
m = X.shape[1]
parameters = ffnn.two_layer_model(X,y,0.2,5000)
y_pred = ffnn.predict(parameters,X_test)
print(y_pred) 
print(y_test)
accuracy = ffnn.accuracy(y_test,y_pred)
print(accuracy)







