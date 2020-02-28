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
        standard_deviation = DataAnalyser.get_standard_deviation(filtered, mean)

        print(f"Column: {self.header[i]} Count: {count} Missing %: {missing_pct} "
              f"Cardinality: {cardinality} Min: {min_value} Max: {max_value} "
              f"First Quantile: {first_quantile} Third Quantile: {third_quantile} "
              f"Mean: {mean} Median: {median} Standard deviation: {standard_deviation}")

    @staticmethod
    def get_standard_deviation(data, mean):
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

        return math.sqrt((1 / (len(data) - 1)) * sum([(item - mean) ** 2 for item in data]))

    @staticmethod
    def get_mode(data):
        result = {}
        unique = set(data)

        for item in unique:
            count = data.count(item)
            result[item] = count

        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        return result[0], result[1]

    @staticmethod
    def get_unique(data):
        result = []
        for item in data:
            if item not in result:
                result.append(item)

        return result

    @staticmethod
    def get_median(data):
        if len(data) % 2 == 0:
            return sum(data) / len(data)

        data.sort()
        return data[len(data) // 2]

    @staticmethod
    def get_quantile(data):
        # median = DataAnalyser.get_median(data)
        # filtered = [i for i in data if i is not None]
        # filtered.sort()
        # first_quantile = DataAnalyser.get_median([item for item in filtered if item < median])
        # third_quantile = DataAnalyser.get_median([item for item in filtered if item > median])

        # return first_quantile, third_quantile
        sorted_data = sorted(data)
        return sorted_data[round(len(data)*1/4)], sorted_data[round(len(data)*3/4)]

    @staticmethod
    def get_missing(data):
        missing = 0
        for item in data:
            if item is None or item == '':
                missing += 1

        return missing
