from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler


def correlation_between(pandas_frame, column1, column2, method='pearson'):
    '''
    Analyze the correlation between two columns

    :param pandas_frame: Input data frame
    :param column1: column 1 to analyze
    :param column2: column 2 to analyze
    :param method: correlation measuring method
        pearson : standard correlation coefficient
            +1 : Total positive linear correlation
            0  : No linear correlation
            -1 : Total negative linear correlation

        kendall : Kendall Tau correlation coefficient
        spearman : Spearman rank correlation

    :return: correlation value
    '''
    return pandas_frame[column1].corr(pandas_frame[column2], method=None)


def correlation_among(pandas_frame, columns=[], method='pearson'):
    '''
    Analyze correlations among several columns

    :param pandas_frame: Input data frame
    :param method: correlation measuring method
    :return: correlation values
    '''

    columns = list(pandas_frame.columns)
    correlations = {}

    for i in range(len(columns)):
        for j in range(i+1, len(columns)-1):
            correlations[(columns[i],columns[j])] = pandas_frame[columns[i]].corr(pandas_frame[columns[j]], method=method)

    sorted_list = sorted(correlations.items(), key=lambda x: x[1])
    # return correlations
    return sorted_list


def vairiance_score(pandas_frame, columns=[], scaled=False):

    '''
    Give variances attached to each column

    :param scaled: Scaled data using sklearn scales
        min_max   : Transforms features by scaling each feature to 0 - 1 range
        max_abs   : Scale each feature by its maximum absolute value.
        robust    : Removes the median and scales the data according to the quantile range
        [Default None]
    :param pandas_frame: Input frame
    :param columns: Selected columns
    :return: columns with variances
    '''

    if scaled == 'min_max':
        scaler = MinMaxScaler()
        pandas_frame[columns] = scaler.fit_transform(pandas_frame[columns])
    elif scaled == 'max_abs':
        scaler = MaxAbsScaler()
        pandas_frame[columns] = scaler.fit_transform(pandas_frame[columns])
    elif scaled == 'robust':
        scaler = RobustScaler()
        pandas_frame[columns] = scaler.fit_transform(pandas_frame[columns])

    variances = {}
    for column in columns:
        variances[column] = np.var(list(pandas_frame[column]), dtype=float)

    sorted_list = sorted(variances.items(), key=lambda x: x[1])
    #return variances
    return sorted_list