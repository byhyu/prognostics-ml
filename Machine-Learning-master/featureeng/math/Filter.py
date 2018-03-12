import numpy as np
import pandas as pd


def threeSigma(series, threshold=3):
    '''
    Identify anomalies according to three sigma rule

    Three Sigma Rule
    ----------------
    std  = standard deviation of data
    mean = mean of data
    if abs(x - mean) > 3 * std then x is an outlier

    :param threshold: 3 is the default value. Change at your own risk
    :param series: input data array
    :return: Index array of where anomalies are
    '''
    series = np.array(list(series))

    std = np.std(series)  # Standard deviation
    avg = np.average(series)  # Mean

    anomaly_indexes = []
    for i in range(series.size):
        if (series[i] - avg) > threshold * std:
            anomaly_indexes.append(i)

    return anomaly_indexes


def iqr(series, threshold=3):
    '''
    Identify anomalies according to Inner-Quartile Range

    IQR Rule
    ----------------
    Q25 = 25 th percentile
    Q75 = 75 th percentile
    IQR = Q75 - Q25 Inner quartile range
    if abs(x-Q75) > 1.5 * IQR : A mild outlier
    if abs(x-Q75) > 3.0 * IQR : An extreme outlier

    :param series: input data array
    :param threshold: 1.5 mild, 3 extreme
    :return: Index array of where anomalies are
    '''

    series = np.array(list(series))
    q25 = np.percentile(series, 25)
    q75 = np.percentile(series, 75)
    iqr = q75 - q25

    anomaly_indexes = []
    for i in range(series.size):
        if (series[i] - q75) > threshold * iqr:
            anomaly_indexes.append(i)

    return anomaly_indexes


def percentile_based(series, lower, upper):
    '''
    Remove anomalies based on the percentile

    :param series: Input series
    :param lower: Lower percentile as a fraction
    :param upper: Upper percentile as a fraction
    :return: Filtered series
    '''

    series = np.array(list(series))
    q_lower = np.percentile(series, lower * 100)
    q_upper = np.percentile(series, upper * 100)

    anomaly_indexes = []
    for i in range(series.size):
        if series[i] < q_lower or series[i] > q_upper:
            anomaly_indexes.append(i)

    x, p = np.histogram(anomaly_indexes)
    return anomaly_indexes


def filterData(panda_frame, columns, removal_method, threshold):
    # Anomaly index container
    rm_index = []

    # Select anomaly removal type
    if removal_method == "iqr":
        for column in columns:
            series = panda_frame[column]
            anomaly = iqr(series, threshold)
            rm_index.extend(anomaly)
    elif removal_method == "threesigma":
        for column in columns:
            series = panda_frame[column]
            anomaly = iqr(series, threshold)
            rm_index.extend(anomaly)

    # Sort indexes
    rm_index.sort()
    anomaly_series = list(set(rm_index))

    # Remove anomalies
    p_filtered = panda_frame.drop(panda_frame.index[anomaly_series])
    return p_filtered


def filterDataPercentile(panda_frame, columns, lower_percentile, upper_percentile, column_err_threshold, order='under'):
    '''
    Filter anomalies based on

    :param panda_frame: Input data frame
    :param columns: Columns that need to apply filter
    :param lower_percentile: Below this level consider as an anomaly
    :param upper_percentile: Beyond this level consider as an anomaly
    :param column_err_threshold: Per column threshold. If a particular row detects as an anomaly how many columns that
                                needs to show as an anomaly
    :return:
    '''
    # Anomaly index container
    rm_index = []

    for column in columns:
        series = panda_frame[column]
        anomaly = percentile_based(series, lower_percentile, upper_percentile)
        rm_index.extend(anomaly)

    dict = {}
    for i in rm_index:
        if dict.has_key(i):
            dict[i] += 1
        else:
            dict[i] = 1

    if order == 'under':
        anomaly_index = [x for x in dict.keys() if dict[x] <= column_err_threshold]
    elif order == 'above':
        anomaly_index = [x for x in dict.keys() if dict[x] >= column_err_threshold]
    # anomaly_count = [dict[x] for x in dict.keys() if dict[x] > column_err_threshold]
    #
    #
    # plt.stem(anomaly_index, anomaly_count)
    # plt.legend(['index', 'count'], loc='upper left')
    # plt.title('Anomaly count')
    # plt.show()

    # Remove anomalies
    p_filtered = panda_frame.drop(panda_frame.index[anomaly_index])
    return p_filtered


def filterDataAutoEncoder(panda_frame, reconstruction_error, threshold):
    '''
    :param panda_frame: Input data frame
    :param reconstruction_error: Reconstruction error fromauto encoders
    :param threshold: Anomaly removal threshold
    :return:
    '''
    rm_index = []
    for i in range(len(reconstruction_error)):
        if reconstruction_error[i] > threshold:
            rm_index.append(i)

    p_filtered = panda_frame.drop(panda_frame.index[rm_index])
    return p_filtered


def indices_seperate(panda_frame=pd, column_name=None):
    '''
        Indices at value changing points, For one dimensional array

        :param column_name: Name of the column
        :param panda_frame: Pandas data frame
        :return: Inices array
    '''

    column = []
    if not column_name:
        return

    try:
        column = panda_frame[column_name]
    except KeyError:
        # Key not found exception
        return

    if len(column) == 0:
        # There is nothing to slice
        return

    column = np.array(column)

    # Each index where the value changes
    indices = np.where(column[:-1] <> column[1:])[0]
    indices = np.insert(indices, len(indices), len(column) - 1, axis=0)
    return indices
