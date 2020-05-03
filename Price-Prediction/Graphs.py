import numpy as np
# from temp import *
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def show_model_verification_plot(_data, title):
    _data = np.array(_data)
    plt.plot(np.arange(len(_data)), _data[:, 0], color='r', label='Spėjimas')
    plt.plot(np.arange(len(_data)), _data[:, 1], color='b', label='Tikėtasis')

    plt.title(title)
    plt.xlabel('Spėjimo nr.')
    plt.ylabel('Saulės dėmių skaičiųs')
    plt.legend()
    plt.show()


def show_sunspots_activity_plot(_sunspots):
    """
    Shows sunspots activity diagram
    :param _sunspots: sunspots data
    """
    plt.plot([sunspot[0] for sunspot in _sunspots], [sunspot[1] for sunspot in _sunspots])
    plt.title("Saulės dėmių aktyvumo 1700-2014 m. grafikas")
    plt.xlabel('Metai')
    plt.ylabel('Saulės dėmių sk.')
    plt.show()


def show_input_output_plot(_sunspots):
    """
    Shows 3d scatter plot with input, output data (Assuming using two previous years to predict)
    :param _sunspots: sunspots data
    """
    data = np.array(prepare_data(_sunspots, 2))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])

    ax.set_title('Įvesčių ir išvesčių trimatė diagrama')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    plt.show()


def show_errors_histogram(_errors):
    plt.hist(_errors, weights=np.ones(len(_errors)), bins=10, ec='black')

    plt.title('Prognozės klaidų histograma')
    plt.xlabel('Paklaida')
    plt.ylabel('Dažnis')
    plt.show()


def show_errors_plot(_errors):
    plt.plot(_errors)

    plt.title('Prognozės klaidų diagrama')
    plt.xlabel('Prognozė')
    plt.ylabel('Paklaida')
    plt.show()


# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
def prepare_data(_data, _number_of_columns):
    _x = []
    _y = []

    for i in range(_number_of_columns, len(_data)):
        temp_array = []
        for j in range(_number_of_columns, 0, -1):
            temp_array.append(_data[i - j][1])

        _x.append(temp_array)
        _y.append(_data[i][1])

    x = list(_x)
    y = list(_y)

    for i, number in enumerate(y):
        x[i].append(number)

    return x


def read_sunspots(file_name):
    with open(file_name, 'r') as f:
        all_sunspots = [row.replace('\n', '').split('\t') for row in f.readlines()]
        return [[int(sunspot[0]), int(sunspot[1])] for sunspot in all_sunspots]


# sunspots = read_sunspots('sunspot.txt')
# # show_input_output_plot(sunspots)
#
# data_min = min(np.array(sunspots)[:, 1])
# data_max = max(np.array(sunspots)[:, 1])
#
# dataset = prepare_data(sunspots, 2)
#
# training_set = dataset[:200]
# validation_set = dataset[200:]
#
# n_inputs = len(training_set[0]) - 1
# n_outputs = 1
#
# network = NeuralNetwork(n_inputs, 1, n_outputs, data_min, data_max)
# network.train(training_set, 0.01, 10000)
#
# MSE, MAD, errors, predictions = network.validate(validation_set)
#
# # show_errors_histogram(errors)
# # show_errors_plot(errors)
# show_model_verification_plot(predictions, title='Modelio verifikacija 1900-2014 metais')
# print(f"MSE: {MSE} MAD: {MAD}")

# np.random.seed(1)
# network = initialize_network(n_inputs, 1, n_outputs)
# print(network)
# print("\n\n\n")
# train_network(network, dataset, 0.01, 10000, n_outputs)
# print(network)
# errors = 0
# max_error = 0
# for row in x[200:]:
#     prediction = predict(network, row)
#     prediction = unnormalize_number(prediction, min_number, max_number)
#     expected_output = unnormalize_number(row[-1], min_number, max_number)
#     # print('Expected=%d, Got=%d' % (row[-1], prediction))
#     error = np.abs(expected_output - prediction)
#     if error >= max_error:
#         max_error = error
#     errors += (expected_output - prediction)**2
#     print(f"Expected={expected_output} Got={prediction}")
#
# print(f"Errors sum: {errors/len(x[200:])} Errors average: {errors / len(x[200:])} Max error: {max_error}")

# plt.plot([sunspot[0] for sunspot in sunspots], [sunspot[1] for sunspot in sunspots])
# plt.title("Saulės dėmių aktyvumo 1700-2014 m. grafikas")
# plt.xlabel('Metai')
# plt.ylabel('Saulės dėmių sk.')
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # for i in range(columns):
# #     ax.scatter(x[:, i], y)
#
# # ax.scatter(x[:, 0], x[:, 1], x[:, 2], x[:, 3], y)
# ax.scatter(P[:, 0], P[:, 1], T)
# plt.show()
#
# import neurolab as nl
#
# # Logical &
# input = [[0, 0], [0, 1], [1, 0], [1, 1]]
# target = [[0], [0], [0], [1]]
#
# # Create net with 2 inputs and 1 neuron
# net = nl.net.newp([[0, 1], [0, 1]], 1)
#
# # train with delta rule
# # see net.trainf
# error = net.trainf(input, target, epochs=100, show=10, lr=0.1)
#
# # Plot results
# import pylab as pl
#
# pl.plot(error)
# pl.xlabel('Epoch number')
# pl.ylabel('Train error')
# pl.grid()
# pl.show()
