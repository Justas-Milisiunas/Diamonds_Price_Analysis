from NeuralNetwork import NeuralNetwork
import numpy as np
from Graphs import *
import matplotlib.pyplot as plt


def read_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        return [line.replace('\n', '') for line in lines]


def parse_data(data):
    parsed_data = []
    for row in data:
        row_parsed = []
        cols = row.split(',')
        for col in cols:
            row_parsed.append(float(col))

        parsed_data.append(row_parsed)

    return np.array(parsed_data)[:, 1:]


def denormalize_prediction(number, min_val, max_val):
    return number * (max_val - min_val) + min_val


# tt = read_data('normalized_data.csv')
#
# new_lines = []
# for line in tt:
#     temp = line.split(',')
#     new_line = temp[0] + ',' + temp[1] + ',' + temp[2] + ',' + temp[3] + ',' + temp[4] + ',' + temp[5] + ',' + temp[6]\
#                + ',' + temp[8] + ',' + temp[9] + ',' + temp[10] + ',' + \
#                temp[7] + '\n'
#     new_lines.append(new_line)
#
# with open('data.csv', 'w') as f:
#     f.writelines(new_lines)

np.random.seed(1)

data = read_data('data.csv')
data = parse_data(data)

np.random.shuffle(data)

verifications_data = []
for i in range(10):
    start_index = int(len(data) / 10) * i
    end_index = start_index + int(len(data) / 10)
    data_set = data[start_index:end_index]
    n_inputs = len(data_set[0]) - 1
    n_outputs = 1
    network = NeuralNetwork(n_inputs, 1, n_outputs)
    progress = network.train(data_set, 0.1, 200, print_info=False)
    MSE, MAD, errors, predictions = network.validate(data_set, print_info=False)
    print(f"{i} MSE: {MSE} MAD: {MAD}")
    verifications_data.append([MSE, MAD, errors, predictions])

mse_sum = np.mean(np.array(verifications_data)[:, 0])
mad_mean = np.mean(np.array(verifications_data)[:, 1])
print(f"AVG MSE: {mse_sum} AVG MAD: {mad_mean}")
# plt.plot(np.arange(len(progress)), progress)
# plt.show()

# rr = 0
# for item in predictions:
#     expected = denormalize_prediction(item[0], 326, 18800)
#     prediction = denormalize_prediction(item[1], 326, 18800)
#
#     rr += np.abs(expected - prediction)
#     print(f"Expected: {expected} Prediction: {prediction}")

# print(f"Vidurkis: {rr/len(predictions)} MAD: {MAD}")
# show_model_verification_plot(predictions[:200], title='Modelio verifikacija')

# dt = [1, 2, 3, 4, 5, 6, 7]
# print(norm(dt))
# network = NeuralNetwork()
