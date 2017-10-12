import numpy as np
import misc


class NN3Layer(object):

    def __init__(self, n_inputs, n_nodes, n_outputs):
        print("Setting up 3 Layer NN with {} inputs, {} nodes and {} outputs\n".format(n_inputs, n_nodes, n_outputs))
        self.n_nodes = n_nodes
        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        self.node_bias = np.random.randn(n_nodes, 1)
        self.output_bias = np.random.randn(n_outputs, 1)
        self.input_node_weights = np.random.randn(n_nodes, n_inputs)
        self.node_output_weights = np.random.randn(n_outputs, n_nodes)


    def compute_node_output(self, input):
        """Compute the output of the node layer, i.e. activation f(apply w'X + b) using the dot product."""
        a = np.zeros(self.n_nodes)
        i = 0
        for weights, bias in zip(self.input_node_weights, self.node_bias):
            z = np.dot(input, weights) + bias
            a[i] = misc.sigmoid(z)
            i += 1
        return a

    def compute_final_output(self, input):
        """Compute the final output, applying activation f(w'X + b) where X is the output from the previous layer"""
        i = 0
        a = np.zeros(self.n_outputs)
        for weights, bias in zip(self.node_output_weights, self.output_bias):
            z = np.dot(input, weights) + bias
            a[i] = misc.tanh(z)
            i += 1
        return a

    def predict(self, input):
        """Predict a 'y' given an input X"""
        a_1 = self.compute_node_output(input)
        a_2 = self.compute_final_output(a_1)
        return a_2

    def compute_error(self, input, y_real):
        """Compute the error of the function by taking squared difference"""
        y_guess = self.predict(input)
        #print("Target: {}".format(y_real))
        #print("Predicted: {}".format(y_guess))
        error = np.dot((y_real - y_guess), (y_real - y_guess))
        #print("Square Error: {}".format(error))
        #print("\n")
        return error

    def gradient_descent_onetime(self, training_input, training_output, step_size):
        """Implement basic gradient descent (online learning)"""
        print("Applying Online Gradient Descent")
        error = self.compute_error(training_input, training_output)
        #print("Initial Error: {}".format(error))
        input_node_w_errors = self.compute_delta_error(self.input_node_weights, training_input, training_output,
                                                     step_size, error)
        node_output_w_erros = self.compute_delta_error(self.node_output_weights, training_input, training_output,
                                                       step_size, error)
        node_b_errors = self.compute_delta_error(self.node_bias, training_input, training_output,
                                                       step_size, error)
        output_b_errors = self.compute_delta_error(self.output_bias, training_input, training_output,
                                                       step_size, error)
        self.input_node_weights += input_node_w_errors
        self.node_output_weights += node_output_w_erros
        self.node_bias += node_b_errors
        self.output_bias += output_b_errors
        error = self.compute_error(training_input, training_output)
        print("New Error: {}".format(error))


    def compute_delta_error(self, parameters, training_input, training_output, step_size, error):
        delta_errors = np.zeros_like(parameters)
        for i in range(0, len(parameters)):
            for j in range(0, len(parameters[0])):
                parameter = parameters[i][j]
                parameters[i][j] += step_size
                if error - self.compute_error(training_input, training_output) > 0:
                    delta_errors[i][j] = step_size
                else:
                    delta_errors[i][j] = -step_size
                parameters[i][j] = parameter
        return delta_errors










