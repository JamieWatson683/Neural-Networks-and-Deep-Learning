import neural_net
import numpy as np


x = [3]
y = [np.sin(x[0])]
net = neural_net.NN3Layer(len(x), 3, len(y))


for i in range(0,100):
    net.gradient_descent_onetime(np.array(x), np.array(y), 0.01)





#net.gradient_descent_onetime(np.array(x), np.array(y), 0.01)

