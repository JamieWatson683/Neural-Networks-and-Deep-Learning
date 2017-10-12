import neural_net
import numpy as np

x = [0.7]
y = [x[0]/2.0]

net = neural_net.NN3Layer(len(x), 5, len(y))
net.gradient_descent_onetime(np.array(x), np.array(y), 0.01)

