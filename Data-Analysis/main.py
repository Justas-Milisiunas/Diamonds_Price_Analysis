from diamond import Diamond
from data_analyser import DataAnalyser


def read_data(file_name):
    _diamonds = []
    file = open(file_name, 'r')
    _header = file.readline().split(',')

    while True:
        line = file.readline()

        if not line:
            return _diamonds, _header

        new_diamond = Diamond().parse(line)
        _diamonds.append(new_diamond)


diamonds, header = read_data('diamonds.csv')

continuous_columns = [1, 5, 6, 7, 8, 9, 10]
categorical_columns = [2, 3, 4]

data_analyser = DataAnalyser(diamonds, header, continuous_columns, categorical_columns)
data_analyser.attributes_analysis()
