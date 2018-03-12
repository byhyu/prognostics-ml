import pandas as pd
from featureeng.math import Features
from featureeng.parser import Text


class Frame:
    def __init__(self, data):
        if isinstance(data, str):
            # File path
            self._pd_frame = pd.read_csv(data)
        elif isinstance(data, pd.core.frame.DataFrame):
            # Panda frame
            self._pd_frame = data

    def get_panda_frame(self):
        '''
        Return modified pandas frame

        :return: pandas frame
        '''
        return self._pd_frame

    def get_column_names(self):
        '''
        Return column names of the given frame

        :return: list
        '''
        return list(self._pd_frame.columns)

    def get_column(self, column_name):
        '''
        Get column by column name

        if column name does not exist, retun empty list

        :param column_name: Column name
        :return: pandas frame
        '''
        if column_name in list(self._pd_frame.columns):
            return self._pd_frame[column_name]
        return []

    def get_row(self, index):
        '''
        Get row by index

        :param index: row number
        :return:
        '''
        return self._pd_frame.iloc[index, :]

    def get_values(self, column_names):
        """
        Get values as an array

        if column(s) not in the data frame, return empty list

        :param column_names: null, string, list of strings
            null            : Return all the data
            string          : return specific column
            list of strings : return columns
        :return: return data as array
        """
        if column_names == None:
            return self._pd_frame.get_values()
        elif type(column_names) == type([]):
            return self._pd_frame[column_names].get_values()
        elif type(column_names) == type(''):
            return self._pd_frame[column_names].get_values()
        else:
            return []

    def add_column(self, column_name, series):
        '''
        Add column data to the frame

        :param column_name: Column name
        :param series: Pandas series
        :return: None
        '''
        self._pd_frame[column_name] = pd.Series(series, index=self._pd_frame.index)

    def append_row(self, row):
        '''
        Append data row to the data frame

        :param row: data row in list format
        :return: None
        '''
        self._pd_frame = self._pd_frame.append(row, ignore_index=True)

    def remove_columns(self, column_names=[]):
        '''
        Remove column names form the data frame

        :param column_names: list of column names
        :return: None
        '''
        for column in column_names:
            if column in list(self._pd_frame.columns):
                del self._pd_frame[column]

    def remove_rows(self, indices=[]):
        '''
        Remove row(s) by index

        :param indices: list of integers
        :return: null
        '''
        self._pd_frame = self._pd_frame.drop(self._pd_frame.index[indices])

    def apply_moving_average(self, input_column, dest_column=None, row_range=(0, None), window=5):
        '''
        Add moving average as another column

        :param input_column: Required column to add feature engineering
        :param dest_column: Destination column name
        :param row_range: Range of rows that need to modify
        :param window: Window size of the calculation takes part
        :return: None
        '''
        if dest_column == None:
            dest_column = input_column + '_ma_' + str(window)

        full_series = list(self._pd_frame[input_column])
        filtered_series = full_series[row_range[0]:row_range[1]]
        result = Features.moving_average(series=filtered_series, window=window, default=True)
        full_series[row_range[0]: row_range[1]] = result
        self.add_column(column_name=dest_column, series=full_series)

    def apply_moving_k_closest_average(self, input_column, dest_column=None, row_range=(0, None), window=5, kclosest=3):
        '''
        Apply moving k closest average as another column

        :param input_column: Required column to add feature engineering
        :param dest_column: Destination column name
        :param row_range: Range of rows that need to modify
        :param window: Window size of the calculation takes part
        :param kclosest: k number of closest values to the recent occurrence including itself
        :return: None
        '''
        if dest_column == None:
            dest_column = input_column + '_kca_' + str(window)

        full_series = list(self._pd_frame[input_column])
        filtered_series = full_series[row_range[0]:row_range[1]]
        result = Features.moving_k_closest_average(series=filtered_series, window=window, kclosest=kclosest,
                                                   default=True)
        full_series[row_range[0]: row_range[1]] = result
        self.add_column(column_name=dest_column, series=full_series)

    def apply_moving_median_centered_average(self, input_column, dest_column=None, row_range=(0, None), window=5,
                                             boundary=1):
        '''
        Apply moving median centered average as another column

        :param input_column: Required column to add feature engineering
        :param dest_column: Destination column name
        :param row_range: Range of rows that need to modify
        :param window: Window size of the calculation takes part
        :param boundary: number of values that need to be removed from both ends of the sorted window
        :return: None
        '''
        if dest_column == None:
            dest_column = input_column + '_mmca_' + str(window)

        full_series = list(self._pd_frame[input_column])
        filtered_series = full_series[row_range[0]:row_range[1]]
        result = Features.moving_median_centered_average(series=filtered_series, window=window, boundary=boundary,
                                                         default=True)
        full_series[row_range[0]: row_range[1]] = result
        self.add_column(column_name=dest_column, series=full_series)

    def apply_moving_weighted_average(self, input_column, dest_column=None, row_range=(0, None), window=5,
                                      weights=[1, 2, 3, 4, 5]):
        '''
        Apply moving weighted average as another column

        :param input_column: Required column to add feature engineering
        :param dest_column: Destination column name
        :param row_range: Range of rows that need to modify
        :param window: Window size of the calculation takes part
        :param weights: list of integers
        :return: None
        '''
        if dest_column == None:
            dest_column = input_column + '_mwa_' + str(window)

        full_series = list(self._pd_frame[input_column])
        filtered_series = full_series[row_range[0]:row_range[1]]
        result = Features.moving_weighted_average(series=filtered_series, window=window, weights=weights, default=True)
        full_series[row_range[0]: row_range[1]] = result
        self.add_column(column_name=dest_column, series=full_series)

    def apply_moving_threshold_average(self, input_column, dest_column=None, row_range=(0, None), window=5,
                                       threshold=0.0):
        '''
        Apply moving threshold as another column

        :param input_column: Required column to add feature engineering
        :param dest_column: Destination column name
        :param row_range: Range of rows that need to modify
        :param window: Window size of the calculation takes part
        :param threshold: double value
        :return: None
        '''
        if dest_column == None:
            dest_column = input_column + '_mta_' + str(window)

        full_series = list(self._pd_frame[input_column])
        filtered_series = full_series[row_range[0]:row_range[1]]
        result = Features.moving_threshold_average(series=filtered_series, window=window, threshold=threshold,
                                                   default=True)
        full_series[row_range[0]: row_range[1]] = result
        self.add_column(column_name=dest_column, series=full_series)

    def apply_moving_median(self, input_column, dest_column=None, row_range=(0, None), window=5):
        '''
        Add moving median as another column

        :param input_column: Required column to add feature engineering
        :param row_range: Range of rows that need to modify
        :param window: Window size of the calculation takes part
        :param dest_column: Destination column name
        :return: None
        '''

        if dest_column == None:
            dest_column = input_column + '_mm_' + str(window)

        full_series = list(self._pd_frame[input_column])
        filtered_series = full_series[row_range[0]:row_range[1]]
        result = Features.moving_median(series=filtered_series, window=window, default=True)
        full_series[row_range[0]: row_range[1]] = result
        self.add_column(column_name=dest_column, series=full_series)

    def apply_moving_std(self, input_column, dest_column=None, row_range=(0, None), window=5):
        '''
        Add moving standard deviation as another column

        :param input_column: Required column to add feature engineering
        :param row_range: Range of rows that need to modify
        :param window: Window size of the calculation takes part
        :param dest_column: Destination column name
        :return: None
        '''
        if dest_column == None:
            dest_column = input_column + '_std_' + str(window)

        full_series = list(self._pd_frame[input_column])
        filtered_series = full_series[row_range[0]:row_range[1]]
        result = Features.moving_standard_deviation(series=filtered_series, window=window, default=True)
        full_series[row_range[0]: row_range[1]] = result
        self.add_column(column_name=dest_column, series=full_series)

    def apply_moving_variance(self, input_column, dest_column=None, row_range=(0, None), window=5):
        '''
                Add moving variance as another column

                :param input_column: Required column to add feature engineering
                :param row_range: Range of rows that need to modify
                :param window: Window size of the calculation takes part
                :param dest_column: Destination column name
                :return: None
                '''
        if dest_column == None:
            dest_column = input_column + '_var_' + str(window)

        full_series = list(self._pd_frame[input_column])
        filtered_series = full_series[row_range[0]:row_range[1]]
        result = Features.moving_variance(series=filtered_series, window=window, default=True)
        full_series[row_range[0]: row_range[1]] = result
        self.add_column(column_name=dest_column, series=full_series)

    def apply_moving_probability(self, input_column, dest_column=None, row_range=(0, None), window=10, no_of_bins=5):
        '''
        Apply moving probability as another columnn

        :param input_column: Required column to add feature engineering
        :param dest_column: Destination column name
        :param row_range: Range of rows that need to modify
        :param window: Window size of the calculation takes part
        :param no_of_bins: Number of discrete levels
        :return: None
        '''
        if dest_column == None:
            dest_column = input_column + '_mprob_' + str(window) + '_' + str(no_of_bins)

        full_series = list(self._pd_frame[input_column])
        filtered_series = full_series[row_range[0]:row_range[1]]
        result = Features.moving_probability(series=filtered_series, window=window, no_of_bins=no_of_bins, default=True)
        full_series[row_range[0]: row_range[1]] = result
        self.add_column(column_name=dest_column, series=full_series)

    def apply_moving_entropy(self, input_column, dest_column=None, row_range=(0, None), window=10, no_of_bins=5):
        '''
        Apply moving entropy as another column

        :param input_column: Required column to add feature engineering
        :param dest_column: Destination column name
        :param row_range: Range of rows that need to modify
        :param window: Window size of the calculation takes part
        :param no_of_bins: Number of discrete levels
        :return: None
        '''

        if dest_column == None:
            dest_column = input_column + '_mentr_' + str(window) + '_' + str(no_of_bins)

        full_series = list(self._pd_frame[input_column])
        filtered_series = full_series[row_range[0]:row_range[1]]
        result = Features.moving_entropy(series=filtered_series, window=window, no_of_bins=no_of_bins, default=True)
        full_series[row_range[0]: row_range[1]] = result
        self.add_column(column_name=dest_column, series=full_series)

    def process_text(self, input_column, case='normal', remove_chars=[]):
        '''
        Apply certain operations to the string in a particular column
        -Case changes
        -Removing characters

        :param input_column:
        :param case:
            normal
            upper
            lower
        :param remove_chars: list of characters that need to be removed from each string
        :return: None
        '''
        column = self.get_column(input_column)
        if len(column) == 0:
            print input_column + " not found"
            return
        full_series = list(column)
        self.add_column(column_name=input_column, series=Text.processText(series=full_series, case=case,
                                                                          remove_chars=remove_chars))

    def look_back(self, input_column, steps=1):
        column = list(self.get_column(column_name=input_column))
        if len(column) == 0:
            return

        for i in range(steps):
            header = input_column + '(t+' + str(i + 1) + ')'
            series = column[i + 1:]
            series.extend([0] * (i + 1))
            self.add_column(column_name=header, series=series)

    def sort(self, column, reset_index=False):
        '''
        Sort data frame according to a given column name

        :param column: sort column
        :param reset_index: True : reset index
                            False: maintain original index
        :return:
        '''
        self._pd_frame = self._pd_frame.sort_values(by=column)
        self._pd_frame = self._pd_frame.reset_index(drop=reset_index)

    def sort_column_names(self):
        '''
        Sort column names

        :return: None
        '''
        column_names = list(self._pd_frame.columns)
        column_names.sort()
        self._pd_frame = self._pd_frame[column_names]

    def order_columns(self, column_names):
        '''
        Order columns in given order

        :param column_names: column name order
        :return: None
        '''
        self._pd_frame = self._pd_frame[column_names]

    def display(self, column_names=[]):
        '''
        Display data frame

        :param column_names: list of required column names to be displayed
        :return:
        '''
        if len(column_names) == 0:
            print self._pd_frame
        else:
            print self._pd_frame[column_names]

    def save_file(self, file_name='out.csv'):
        '''
        Save file

        :param file_name: destination file name
        :return: None
        '''
        self._pd_frame.to_csv(file_name, index=False)
