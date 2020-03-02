import math
import numpy as np


class DataAnalyser:
    def __init__(self, _data, _header, _continuous, _categorical):
        self.data = _data
        self.header = _header
        self.continuous = _continuous
        self.categorical = _categorical

    def attributes_analysis(self):
        all_data = []
        for item in self.data:
            all_data.append(list(item.__dict__.values()))

        for i in range(len(all_data[0])):
            # if not isinstance(all_data[0][i], str):
            items = [row[i] for row in all_data]
            self.analyse_column(i, items)

    def analyse_column(self, i, column):
        if i in self.continuous:
            self.analyse_continuous(i, column)
        elif i in self.categorical:
            self.analyse_categorical(i, column)

    def analyse_categorical(self, i, column):
        """
        Analyses categorical attributes by getting: count, missing count, cardinality, mode,
        mode frequency and percentage, second mode, second mode frequency and percentage
        :param i: column index
        :param column: column data
        :return:
        """
        count = len(column)
        missing = DataAnalyser.get_missing(column)
        missing_pct = missing * 100 / count

        filtered = [i for i in column if i is not None or i == '']
        cardinality = len(DataAnalyser.get_unique(filtered))
        (mode, mode_frequency), (mode2, mode2_frequency) = DataAnalyser.get_mode(filtered)
        mode_pct = mode_frequency * 100 / count
        mode2_pct = mode2_frequency * 100 / count

        print(f"Column: {self.header[i]} Count: {count} Missing %: {missing_pct} Cardinality: {cardinality} "
              f"Mode: {mode} Mode frequency: {mode_frequency} Mode %: {mode_pct} "
              f"Mode 2: {mode2} Mode 2 frequency: {mode2_frequency} Mode 2 %: {mode2_pct}")

    def analyse_continuous(self, i, column):
        """
        Analyses continuous attributes by getting: count, missing count, cardinality,
        min value, max value, first and third quantile, average value, median,  standard deviation
        :param i: column index
        :param column: column data
        :return:
        """
        count = len(column)
        missing = DataAnalyser.get_missing(column)
        missing_pct = missing * 100 / count

        filtered = [i for i in column if i is not None]
        cardinality = len(DataAnalyser.get_unique(filtered))

        min_value = min(filtered)
        max_value = max(filtered)

        mean = sum(filtered) / count
        median = DataAnalyser.get_median(filtered)

        first_quantile, third_quantile = DataAnalyser.get_quantile(filtered)
        standard_deviation = DataAnalyser.get_standard_deviation(filtered)

        print(f"Column: {self.header[i]} Count: {count} Missing %: {missing_pct} "
              f"Cardinality: {cardinality} Min: {min_value} Max: {max_value} "
              f"First Quantile: {first_quantile} Third Quantile: {third_quantile} "
              f"Mean: {mean} Median: {median} Standard deviation: {standard_deviation}")

    def get_covariation_matrix(self):
        all_data = []
        for item in self.data:
            all_data.append(list(item.__dict__.values()))

        filtered_data = {}
        for i in range(len(all_data[0])):
            # if not isinstance(all_data[0][i], str):
            if i not in self.continuous:
                continue
            filtered_data[self.header[i]] = [row[i] for row in all_data]

        for key in filtered_data.keys():
            for next_key in filtered_data.keys():
                if key == next_key:
                    continue
                print(
                    f"{key} {next_key} covariation: {DataAnalyser.get_covariation_between(filtered_data[key], filtered_data[next_key])}")

    def save_normalized_data(self, file_name):
        all_data = []
        for item in self.data:
            all_data.append(list(item.__dict__.values()))

        filtered_data = {}
        for i in range(len(all_data[0])):
            # if not isinstance(all_data[0][i], str):
            if i not in self.continuous:
                continue
            filtered_data[self.header[i]] = [row[i] for row in all_data]

        for column in filtered_data.keys():
            min_value = min(filtered_data[column])
            max_value = max(filtered_data[column])
            for i, item in enumerate(filtered_data[column]):
                new_value = (item - min_value) / (max_value - min_value) * (1 - 0) + 0
                filtered_data[column][i] = new_value
        with open(file_name, 'w') as file:
            header_row = ''
            for column in self.header:
                header_row += column + ','

            file.write(header_row[0:-1])
            lines = []
            for i in range(len(self.data)):
                item = ''
                for column in self.header:
                    if column in filtered_data.keys():
                        item += str(filtered_data[column][i]) + ','
                    else:
                        item += str(getattr(self.data[i], column)) + ','

                lines.append(item[:-1] + '\n')

            file.writelines(lines)

    @staticmethod
    def get_covariation_between(data1, data2):
        if len(data1) != len(data2):
            raise Exception('Length must be the same')

        mean1 = sum(data1) / len(data1)
        mean2 = sum(data2) / len(data2)

        _sum = 0
        for i in range(len(data1)):
            _sum += (data1[i] - mean1) * (data2[i] - mean2)

        return (1 / (len(data1) - 1)) * _sum

    @staticmethod
    def get_standard_deviation(data):
        """
        Standard deviation calculation
        source: https://en.wikipedia.org/wiki/Standard_deviation#/media/File:Metabolic_rates_for_northern_fulmars.svg
        :param data: data
        :param mean: mean of data
        :return: calculated standard deviation
        """
        # sum = 0
        # for item in data:
        #     sum += (item - mean) ** 2
        mean = sum(data) / len(data)
        return math.sqrt((1 / (len(data) - 1)) * sum([(item - mean) ** 2 for item in data]))

    @staticmethod
    def get_mode(data):
        """

        :param data:
        :return:
        """
        result = {}
        unique = set(data)

        for item in unique:
            count = data.count(item)
            result[item] = count

        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        return result[0], result[1]

    @staticmethod
    def get_unique(data):
        """
        Returns unique values list from data
        :param data: Data
        :return: Unique values list
        """
        result = []
        for item in data:
            if item not in result:
                result.append(item)

        return result

    @staticmethod
    def get_median(data):
        """
        Calculates data median
        :param data: Data
        :return: Median
        """
        if len(data) % 2 == 0:
            return sum(data) / len(data)

        data.sort()
        return data[len(data) // 2]

    @staticmethod
    def get_quantile(data):
        """
        Calculates first and third quantile
        :param data: Data
        :return: First and third quantile
        """
        # median = DataAnalyser.get_median(data)
        # filtered = [i for i in data if i is not None]
        # filtered.sort()
        # first_quantile = DataAnalyser.get_median([item for item in filtered if item < median])
        # third_quantile = DataAnalyser.get_median([item for item in filtered if item > median])

        # return first_quantile, third_quantile
        sorted_data = sorted(data)
        return sorted_data[round(len(data) * 1 / 4)], sorted_data[round(len(data) * 3 / 4)]

    @staticmethod
    def get_missing(data):
        """
        Counts missing values (None)
        :param data: Data
        :return: Missing values count
        """
        missing = 0
        for item in data:
            if item is None or item == '':
                missing += 1

        return missing
