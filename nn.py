import numpy as np

def initialize_parameters(size_input, size_hidden, size_output):    
    w1 = np.random.randn(size_hidden,size_input)*0.01
    b1 = np.zeros((size_hidden,1))
    w2 = np.random.randn(size_output,size_hidden)*0.01
    b2 = np.zeros((size_output,1))    
    parameters = {
        "w1":w1,
        "b1":b1,
        "w2":w2,
        "b2":b2,
    }
    return parameters

def feed_forward(parameters,x):
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    z1 = np.dot(w1,x) + b1
    a1 = relu(z1)
    z2 = np.dot(w2,a1) + b2
    a2 = sigmoid(z2)
    ff = {
        "z1":z1,
        "z2":z2,
        "a1":a1,
        "a2":a2,
        "w2":w2        
    }
    return ff

def back_prop(ff,y,x):
    m = y.shape[1]
    z1 = ff["z1"]
    z2 = ff["z2"]
    z2 = ff["z2"]
    a1 = ff["a1"]
    a2 = ff["a2"]
    w2 = ff["w2"]
    dz2 = a2 - y
    dw2 = (1/m)*np.dot(dz2,a1.T)
    db2 = (1/m)*np.sum(dz2,axis = 1, keepdims = True)
    da1 = np.dot(w2.T,dz2)
    dz1 = da1*relu_grad(z1)
    dw1 = (1/m)*np.dot(dz1,x.T)
    db1 = (1/m)*np.sum(dz1, axis = 1, keepdims = True)
    grads = {
        "dw1":dw1,
        "dw2":dw2,
        "db1":db1,
        "db2":db2
    }
    return grads

def update_parameters(LEARNING_RATE,parameters,grads):
    w1 = parameters["w1"]
    w2 = parameters["w2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    dw1 = grads["dw1"]
    dw2 = grads["dw2"]
    db1 = grads["db1"]
    db2 = grads["db2"]
    
    w1 = w1 - LEARNING_RATE*dw1
    w2 = w2 - LEARNING_RATE*dw2
    b1 = b1 - LEARNING_RATE*db1
    b2 = b2 - LEARNING_RATE*db2

    updated_parameters = {
        "w1":w1,
        "w2":w2,
        "b1":b1,
        "b2":b2,
    }
    return updated_parameters

def sigmoid(a):
    return 1/(1 + np.exp(-1*a))

def relu(a):
    for i in range(0,a.shape[0]):
        for j in range(0,a.shape[1]):
            if a[i][j]>0:
                a[i][j] = a[i][j]
            else:
                a[i][j] = 0
    return a

def sigmoid_grad(a):
    return sigmoid(a)*(sigmoid(a)-1)

def relu_grad(a):
    for i in range(0,a.shape[0]):
        for j in range(0,a.shape[1]):
            if a[i][j]>0:
                a[i][j] = 1
            else:
                a[i][j] = 0
    return a

def cost(a2,y):
    m = y.shape[1]
    c = (-1/m)*(np.dot(y,np.log(a2.T))+np.dot(1-y,np.log((1-a2.T))))
    return c

