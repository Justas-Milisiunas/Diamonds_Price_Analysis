from diamond import Diamond
from data_analyser import DataAnalyser
from data_graphs import DataGraphs


def read_data(file_name):
    _diamonds = []
    file = open(file_name, 'r')
    _header = [item.replace('"', '').rstrip('\n') for item in file.readline().split(',')]

    while True:
        line = file.readline()

        if not line:
            return _diamonds, _header

        new_diamond = Diamond().parse(line)
        _diamonds.append(new_diamond)


def filter_data(data):
    filtered_data = []
    for item in data:
        if item.length != 0 and item.width != 0 and item.depth != 0:
            filtered_data.append(item)

    return filtered_data


diamonds, header = read_data('diamonds.csv')

continuous_columns = [1, 5, 6, 7, 8, 9, 10]
categorical_columns = [2, 3, 4]

# data_analyser = DataAnalyser(diamonds, header, continuous_columns, categorical_columns)
# data_analyser.attributes_analysis()

data_graphs = DataGraphs(filter_data(diamonds), header)
# data_graphs.show_histogram('carat', 30)
# data_graphs.show_scatter_graph('depth_pct', 'price')
# data_graphs.show_stacked_bars('cut', 'color')
data_graphs.show_box_plot('price', 'cut')