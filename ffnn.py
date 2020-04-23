import nn
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(72)
def two_layer_model(x,y,LEARNING_RATE,NUM_ITER):

    size_input = x.shape[0]
    size_output = 1
    size_hidden = x.shape[0]
    costs = []
    parameters = nn.initialize_parameters(size_input,size_hidden,size_output)

    for i in range(0,NUM_ITER):        
            ff = nn.feed_forward(parameters,x)
            a2 = ff["a2"]
            cos = nn.cost(a2,y)
            grads = nn.back_prop(ff,y,x)
            updated_parameters = nn.update_parameters(LEARNING_RATE,parameters,grads)
            parameters["w1"] = updated_parameters["w1"]
            parameters["w2"] = updated_parameters["w2"]
            parameters["b1"] = updated_parameters["b1"]
            parameters["b2"] = updated_parameters["b2"]
            if i % 100 == 0:
                print("Cost after iteration {}: {}".format(i, cos))
            if i % 100 == 0:
                costs.append(cos)

    print("Cost after iteration {}: {}".format(i, np.squeeze(cos)))
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(LEARNING_RATE))
    plt.show()

    return updated_parameters

def predict(updated_parameters,x):
    w1 = updated_parameters["w1"]
    w2 = updated_parameters["w2"]
    b1 = updated_parameters["b1"]
    b2 = updated_parameters["b2"]
    z1 = np.dot(w1,x) + b1
    a1 = nn.relu(z1)
    z2 = np.dot(w2,a1) + b2
    a2 = nn.sigmoid(z2)
    for i in range(0,a2.shape[0]):
        for j in range(0,a2.shape[1]):
            if a2[i][j]>0.5:
                a2[i][j] = 1
            else:
                a2[i][j] = 0
    return a2 

def accuracy(y_true, y_pred):
    count = 0
    l = y_true.shape[1]
    for i in range(0,y_true.shape[0]):
        for j in range(0,y_true.shape[1]):
            if(y_true[i][j]==y_pred[i][j]):
                count = count + 1
    return (count/l)*100