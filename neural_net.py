import numpy as np
import misc


class NN3Layer(object):

    def __init__(self, n_inputs, n_nodes, n_outputs):
        self.n_nodes = n_nodes
        self.n_outputs = n_outputs
        self.node_bias = np.random.randn(n_nodes)
        self.output_bias = np.random.randn(n_outputs)
        self.input_node_weights = np.random.randn(n_nodes, n_inputs)
        self.node_output_weights = np.random.randn(n_outputs, n_nodes)

    def compute_node_output(self, input):
        """Compute the output of the node layer, i.e. activation f(apply w'X + b) using the dot product."""
        i = 0
        z = np.zeros(shape=self.n_nodes)
        for weights, bias in zip(self.input_node_weights, self.node_bias):
            z[i] = misc.sigmoid(np.dot(input, weights) + bias)
            i += 1
        return z

    def compute_final_output(self, input):
        """Compute the final output, applying activation f(w'X + b) where X is the output from the previous layer"""
        i = 0
        z = np.zeros(shape=self.n_outputs)
        for weights, bias in zip(self.node_output_weights, self.output_bias):
            z[i] = misc.tanh(np.dot(input, weights) + bias)
            i += 1
        return z

    def predict(self, input):
        """Predict a 'y' given an input X"""
        a_1 = self.compute_node_output(input)
        a_2 = self.compute_final_output(a_1)
        return a_2

    def compute_error(self, input, y_real):
        """Compute the error of the function by taking squared difference"""
        y_guess = self.predict(input)
        error = sum((y_real - y_guess)*(y_real - y_guess))
        return error

    def gradient_descent_onetime(self, training_input, training_output, step_size):
        error = self.compute_error(training_input, training_output)
        delta_errors = [] # convert to np.array
        i = 0
        for weight in self.input_node_weights: # convert to separate function?
            self.input_node_weights[i] += step_size
            delta_error = self.compute_error(training_input, training_output)
            delta_errors.append(error - delta_error)
            self.input_node_weights[i] = weight
            i += 1
        print delta_errors









