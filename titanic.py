import nn
import matplotlib.pyplot as plt

def two_layer_model(x,y,LEARNING_RATE=0.1,NUM_ITER = 1000):

    size_input = x.shape[0]
    size_output = y.shape[0]
    size_hidden = 1

    parameters = initialize_parameters(size_input,size_hidden,size_output)
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]

    for i in range(0,NUM_ITER):    
        ff = feed_forward(parameters,x)
        grads = back_prop(ff,y,x)
        a2 = ff["a2"]
        cost = cost(a2,y)
        updated_parameters = update_parameters(LEARNING_RATE,parameters,grads)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("LEARNING_RATE =" + str(learning_rate))
    plt.show()

    return updated_parameters
