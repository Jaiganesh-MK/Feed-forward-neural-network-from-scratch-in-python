import ffnn 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split

df = pd.read_csv("/home/mk/Desktop/feed_forward_neural_net/titanic.csv",header = None)
y = df[0]
x = df.drop([0],axis = 1)
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 22)
X = X_train.to_numpy()
y = y_train.to_numpy().reshape(-1,1)
parameters = ffnn.two_layer_model(X.T,y.T,0.0075,1000)





