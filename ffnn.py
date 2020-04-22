import nn
import matplotlib.pyplot as plt
import numpy as np

def two_layer_model(x,y,LEARNING_RATE,NUM_ITER):

    size_input = x.shape[0]
    size_output = y.shape[0]
    size_hidden = 1

    parameters = nn.initialize_parameters(size_input,size_hidden,size_output)
    costs = []

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
            print("Cost after iteration {}: {}".format(i, np.squeeze(cos)))
        if i % 100 == 0:
            costs.append(cos)

    print("Cost after iteration {}: {}".format(i, np.squeeze(cos)))
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(LEARNING_RATE))
    plt.show()

    return updated_parameters
