import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def tanh(z):
    return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
