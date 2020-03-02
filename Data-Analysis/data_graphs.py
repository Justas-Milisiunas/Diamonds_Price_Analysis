import matplotlib.pyplot as plt
import numpy as np
from data_analyser import DataAnalyser
import pandas as pd
import seaborn as sn


class DataGraphs:
    def __init__(self, _data, _headers):
        self.header = _headers
        self.columns = []

        all_data = []
        for item in _data:
            all_data.append(list(item.__dict__.values()))

        for i in range(len(all_data[0])):
            self.columns.append([row[i] for row in all_data])

    def show_histogram_graph(self, column_name, bins_count):
        """
        Shows selected column's data histogram divided in given amount of bins
        :param column_name: Column name
        :param bins_count: Bins count
        """
        data = self.columns[self.header.index(column_name)]
        plt.hist(data, weights=np.ones(len(data)) / len(data), ec='k', bins=bins_count)
        plt.ylabel('Frequency')
        plt.xlabel(column_name)
        plt.show()

    def show_scatter_graph(self, column_name, column2_name):
        """
        Shows scatter graph between to selected continuous attributes
        (Uncomment code to see trend line)
        :param column_name: First attribute
        :param column2_name: Second attribute
        """
        data = self.columns[self.header.index(column_name)]
        data2 = self.columns[self.header.index(column2_name)]

        plt.plot(data, data2, 'o')
        # plt.scatter(data, data2)
        z = np.polyfit(data, data2, 1)
        p = np.poly1d(z)
        plt.plot(data, p(data), "r")
        # plt.ylim(0, 12)
        # plt.xlim(0, 12)

        plt.xlabel(column_name)
        plt.ylabel(column2_name)
        # plt.legend()
        plt.show()

    def show_stacked_bars(self, top_column, bottom_column):
        """
        Shows stacked bars graph from two selected categorical attributes data
        :param top_column: Top column
        :param bottom_column: Bottom column
        """
        top_data = self.columns[self.header.index(top_column)]
        bottom_data = self.columns[self.header.index(bottom_column)]

        data = self.format_data(top_data, bottom_data)

        bars_names = list(self.get_unique(top_data).keys())
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        colors_labels = list(self.get_unique(bottom_data).keys())

        color_counter = 1
        labels_counter = 1
        width = 0.5

        normalized_data = self.normalize_data(data)
        bars_sum = np.array(normalized_data[0])

        plt.bar(bars_names, normalized_data[0], width, color=colors[color_counter], label=colors_labels[0])
        for i, bar_data in enumerate(normalized_data[1:]):
            plt.bar(bars_names, bar_data, width, color=colors[color_counter % len(colors)], bottom=bars_sum,
                    label=colors_labels[labels_counter])
            bars_sum += bar_data
            labels_counter += 1
            color_counter += 1

        plt.xlabel(top_column)
        plt.ylabel('%')
        plt.legend(title=bottom_column)
        plt.show()

    def show_box_plot(self, continuous_attribute, categorical_attribute):
        """
        Shows box plot graph between one continuous and one categorical attributes
        :param continuous_attribute: Continuous attribute
        :param categorical_attribute: Categorical attribute
        """
        cont_attr = self.columns[self.header.index(continuous_attribute)]
        cat_attr = self.columns[self.header.index(categorical_attribute)]

        x_labels = list(self.get_unique(cat_attr).keys())

        data = []
        for x in x_labels:
            data.append(self.filter_column(cont_attr, categorical_attribute, x))

        plt.boxplot([bar for bar in data])
        plt.xticks([y + 1 for y in range(len(x_labels))], x_labels)
        plt.ylabel(continuous_attribute)
        plt.xlabel(categorical_attribute)
        plt.show()

    def show_correlation_matrix(self, columns):
        """
        Shows correlation matrix plot between given column indexes
        :param columns: Columns indexes
        """
        data = {}
        for column in columns:
            data[self.header[column]] = self.columns[column]

        df = pd.DataFrame(data)

        corr = df.corr()
        corr.style.background_gradient(cmap='coolwarm')

        sn.heatmap(corr, annot=True)
        plt.show()

    def show_scatter_matrix(self):
        """
        Shows data's scatter matrix graph
        """
        sn.set()

        df = pd.read_csv('diamonds.csv', usecols=lambda col: col not in ['id'])
        sn.pairplot(df, plot_kws={'edgecolor': 'none', 's': 5})
        plt.show()

    def filter_column(self, data, condition_column, condition):
        """
        Filter data list by given value
        :param data: Data list
        :param condition_column: Condition column name
        :param condition: Condition value
        :return: Filtered data
        """
        result = []
        condition_index = self.header.index(condition_column)
        for i, item in enumerate(data):
            if self.columns[condition_index][i] == condition:
                result.append(item)

        return result

    def format_data(self, data, data2):
        """
        Formats data for stacked bar graph
        :param data: first column data
        :param data2: second column data
        :return: formatted data
        """
        result = self.get_unique(data)

        for i, item in enumerate(data2):
            if item not in result[data[i]].keys():
                result[data[i]][item] = 0

            result[data[i]][item] += 1

        unique_data2 = self.get_unique(data2).keys()
        result_array = []
        for item in unique_data2:
            temp_data = []
            for key in result:
                temp_data.append(result[key][item])

            result_array.append(temp_data)

        return result_array

    @staticmethod
    def get_unique(data):
        """
        Extracts unique values from given data
        :param data: Data
        :return: Unique values array
        """
        result = {}
        for i, item in enumerate(data):
            if item not in result.keys():
                result[item] = {}

        return result

    @staticmethod
    def normalize_data(data):
        """
        Normalizes data for stacked bars graph
        :param data: Data
        :return: Normalized data
        """
        all_sums = [0] * len(data[0])
        for bar in data:
            for i, bar_data in enumerate(bar):
                all_sums[i] += bar_data

        result = []
        for bar in data:
            bar_result = []
            for i, bar_data in enumerate(bar):
                bar_result.append(bar_data * 100 / all_sums[i])

            result.append(bar_result)

        return result
