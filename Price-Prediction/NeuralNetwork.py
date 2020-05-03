from random import seed
from random import random
import numpy as np


class NeuralNetwork:
    def __init__(self, _inputs_count, _hidden_layers_count, _outputs_count, min_value=0, max_value=0):
        self.inputs_count = _inputs_count
        self.hidden_layers_count = _hidden_layers_count
        self.outputs_count = _outputs_count
        self.min_value = min_value
        self.max_value = max_value

        self.network = []
        hidden_layer = [{'weights': list(2 * np.random.random(_inputs_count + 1) - 1)} for i in
                        range(_hidden_layers_count)]
        self.network.append(hidden_layer)
        output_layer = [{'weights': list(2 * np.random.random(_outputs_count + 1) - 1)} for i in range(_outputs_count)]
        self.network.append(output_layer)

    def transfer(self, activation):
        return 1.0 / (1.0 + np.exp(-activation))

    def activate(self, weights, inputs):
        """
        Calcules weighted sum of inputs with given weights
        :param weights: Layer's weights
        :param inputs: Inputs
        :return: Weighted sum of inputs
        """
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]

        return activation

    # Calculate the derivative of an neuron output
    def transfer_derivative(self, output):
        """
        Used formula: f(x) = 1 / (1 + e^(-x))
        :param output: x
        :return: calculated value
        """
        return output * (1.0 - output)

    def forward_propagate(self, row):
        """
        Goes through each network's layer and sums multiplied inputs by layer's weights + bias
        :param row: input data
        :return: calculated output
        """
        inputs = row
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.transfer(activation)
                new_inputs.append(neuron['output'])

            inputs = new_inputs
        return inputs

    def backward_propagate_error(self, expected):
        """
        Calculates and saves each neuron's error
        :param expected: expected output
        """
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            if i != len(self.network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])

            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.transfer_derivative(neuron['output'])

    def update_weights(self, row, l_rate):
        """
        Updates each neuron's weight with it's saved error
        :param row: input data
        :param l_rate: learning rate
        """
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += l_rate * neuron['delta']

    def train(self, dataset, l_rate, n_epoch, print_info=True):
        """
        Manages network's training by getting current prediction,
        calculating error, updating weights based on that error
        :param dataset: train dataset
        :param l_rate: learning rate
        :param n_epoch: number of epochs
        :param print_info: print epoch info
        """
        # normalized_data = self.__normalize_data(dataset)
        epochs_errors = []
        for epoch in range(n_epoch):
            sum_error = 0
            for row in dataset:
                outputs = self.forward_propagate(row)
                # expected = [0 for i in range(n_outputs)]
                # expected[row[-1]] = 1
                expected = [row[-1]]
                sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                self.backward_propagate_error(expected)
                self.update_weights(row, l_rate)

            epochs_errors.append(sum_error)

            if print_info:
                print(f">epoch={epoch}, lrate={l_rate}, error={sum_error}")

        return epochs_errors

    def validate(self, dataset, print_info=False):
        """
        Validates neural network by measuring prediction accuracy
        :param dataset: Validation data set
        :param print_info: Print results to console
        :return: MSE (Mean-Square Error), MAD (Median Absolute Deviation), errors list, predictions list
        """
        errors = 0
        errors_list = []
        expected_prediction = []
        for row in dataset:
            prediction = self.predict(row)
            expected_output = row[-1]

            errors += (expected_output - prediction) ** 2
            errors_list.append(expected_output - prediction)
            expected_prediction.append([expected_output, prediction])
            if print_info:
                print(f"Expected={expected_output} Prediction={prediction}")

        mse = errors / len(dataset)
        if print_info:
            print(f"MSE: {mse}")

        return mse, np.median(np.abs(errors_list)), errors_list, expected_prediction

    def predict(self, row):
        """
        Does prediction from given the input
        :param row: input
        :return: prediction result
        """
        # normalized = self.__normalize_prediction(row)
        outputs = self.forward_propagate(row)
        # denormalized = self.__denormalize_prediction(outputs)
        return outputs[0]

    def __denormalize_data(self, dataset):
        min_value = self.min_value
        max_value = self.max_value

        n_cols = len(dataset[0])
        _data = np.array(dataset).flatten()

        _data = np.array([number * (max_value - min_value) + min_value for number in _data])
        _data = np.reshape(_data, (-1, n_cols))

        return list(_data)

    def __denormalize_prediction(self, prediction):
        return [number * (self.max_value - self.min_value) + self.min_value for number in prediction]

    def __normalize_data(self, dataset, low=0, high=1):
        n_cols = len(dataset[0])
        _data = np.array(dataset).flatten()

        min_number = self.min_value
        max_number = self.max_value

        _data = np.array([(number - min_number) / (max_number - min_number) * (high - low) + low for number in _data])
        _data = np.reshape(_data, (-1, n_cols))

        return list(_data)

    def __normalize_prediction(self, row):
        return [(number - self.min_value) / (self.max_value - self.min_value) for number in row]

# # Initialize a network
# def initialize_network(n_inputs, n_hidden, n_outputs):
#     network = list()
#     tt = list(2*np.random.random(n_inputs + 1) - 1)
#     # hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
#     hidden_layer = [{'weights': tt} for i in range(n_hidden)]
#     network.append(hidden_layer)
#     # output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
#     ttr = list(2*np.random.random(n_outputs + 1) - 1)
#     output_layer = [{'weights': ttr} for i in range(n_outputs)]
#     network.append(output_layer)
#     return network
#
#
# def transfer(activation):
#     return 1.0 / (1.0 + np.exp(-activation))
#
#
# def activate(weights, inputs):
#     activation = weights[-1]
#     for i in range(len(weights) - 1):
#         activation += weights[i] * inputs[i]
#
#     return activation
#
#
# # Calculate the derivative of an neuron output
# def transfer_derivative(output):
#     return output * (1.0 - output)
#
#
# def forward_propagate(network, row):
#     inputs = row
#     for layer in network:
#         new_inputs = []
#         for neuron in layer:
#             activation = activate(neuron['weights'], inputs)
#             neuron['output'] = transfer(activation)
#             new_inputs.append(neuron['output'])
#
#         inputs = new_inputs
#     return inputs
#
#
# def backward_propagate_error(network, expected):
#     for i in reversed(range(len(network))):
#         layer = network[i]
#         errors = list()
#         if i != len(network) - 1:
#             for j in range(len(layer)):
#                 error = 0.0
#                 for neuron in network[i + 1]:
#                     error += (neuron['weights'][j] * neuron['delta'])
#                 errors.append(error)
#         else:
#             for j in range(len(layer)):
#                 neuron = layer[j]
#                 errors.append(expected[j] - neuron['output'])
#
#         for j in range(len(layer)):
#             neuron = layer[j]
#             neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
#
#
# def update_weights(network, row, l_rate):
#     for i in range(len(network)):
#         inputs = row[:-1]
#         if i != 0:
#             inputs = [neuron['output'] for neuron in network[i - 1]]
#         for neuron in network[i]:
#             for j in range(len(inputs)):
#                 neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
#             neuron['weights'][-1] += l_rate * neuron['delta']
#
#
# def train_network(network, train, l_rate, n_epoch, n_outputs):
#     for epoch in range(n_epoch):
#         sum_error = 0
#         for row in train:
#             outputs = forward_propagate(network, row)
#             # expected = [0 for i in range(n_outputs)]
#             # expected[row[-1]] = 1
#             expected = [row[-1]]
#             sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
#             backward_propagate_error(network, expected)
#             update_weights(network, row, l_rate)
#         print(f">epoch={epoch}, lrate={l_rate}, error={sum_error}")
#
#
# def predict(network, row):
#     outputs = forward_propagate(network, row)
#     return round(outputs[0])
#
#
# dataset = [[0, 0, 1, 0],
#            [1, 1, 1, 1],
#            [1, 0, 1, 1],
#            [0, 1, 1, 0]]
# n_inputs = len(dataset[0]) - 1
# n_outputs = 1
#
# np.random.seed(1)
# network = initialize_network(n_inputs, 1, n_outputs)
# print(network)
# print("\n\n\n")
# train_network(network, dataset, 0.01, 10000, n_outputs)
# print(network)
# for row in dataset:
#     prediction = predict(network, row)
#     print('Expected=%d, Got=%d' % (row[-1], prediction))
