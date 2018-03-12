import numpy as np
import pandas as pd
from common import Select
from featureeng.parser import XMLParser
from featureeng import Frame
import os

'''
Processed Dataset
-----------------

Apply feature engineering operations and generate
train and test frames.

Input directory  : csv
Output directory : processed-data
'''


def remaining_usefullifetime(indices=np.array([]), time_series=np.array([])):
    '''
    Calculate remaining useful life
    :param indices: Separations at time series
    :param time_series: Time series values
    :return: Remaining useful time
    '''

    # add final index
    indices = np.insert(indices, len(indices), len(time_series) - 1, axis=0)

    rul = np.array([])

    # generate remaining useful array
    for i in indices:
        rul = np.concatenate((rul, range(time_series[i])[::-1]), axis=0)

    return rul


def binary_classification(indices=np.array([]), time_series=np.array([])):
    '''
    Add binary classification label

    :param indices: Seperation at time series
    :param time_series: Time series values
    :return: Binary classified classes
    '''
    # add final index
    indices = np.insert(indices, len(indices), len(time_series) - 1, axis=0)

    bin_classification = np.array([])

    for i in indices:
        class_label = range(time_series[i])[::-1]
        class_label = [0 if x >= 30 else 1 for x in class_label]
        bin_classification = np.concatenate((bin_classification, class_label), axis=0)

    return bin_classification


def create_train_dataset(file_name='train_FD001.csv'):
    training_frame = Frame('csv/' + file_name)

    # Apply feature engineering for training data set
    # Feature engineering configurations contain in 'feature_engineering.xml' file
    training_frame = XMLParser.apply_feature_eng(training_frame, 'feature_engineering')

    # Seperate machine by unit number
    indices = Select.indices_seperate(feature_name="UnitNumber", data_frame=training_frame)
    time_column = training_frame['Time']

    # Add remaining useful life in cycles
    rul = remaining_usefullifetime(indices=indices, time_series=time_column)
    training_frame['RUL'] = rul

    training_frame.to_csv('test-regression/processed-data/train.csv', index=False)

    del training_frame['RUL']

    # Add binary classification
    label = binary_classification(indices=indices, time_series=time_column)
    training_frame['BIN'] = map(int, label)
    training_frame.to_csv('test-classification/processed-data/train.csv', index=False)


def create_test_dataset(test_file='test_FD001.csv', rul_file='RUL_FD001.csv'):
    testing_frame = Frame('csv/' + test_file)

    '''
    Apply feature engineering for testing data set
    ----------------------------------------------

    Feature engineering configurations contain in 'feature_engineering.xml' file
    Training and testing frames should have same feature engineering process. That is
    why training and testing dataset use the same feature engineering configuration file
    '''

    testing_frame = XMLParser.apply_feature_eng(testing_frame, 'feature_engineering')
    rul_frame = pd.read_csv('csv/' + rul_file)

    # Select seperation points to apply moving operations
    indices = Select.indices_seperate(feature_name="UnitNumber", data_frame=testing_frame)

    filtered_frame = pd.DataFrame(columns=testing_frame.columns)

    # Add last index to the indices
    indices = np.insert(indices, len(indices), len(testing_frame['UnitNumber']) - 1, axis=0)

    # Select lines for test
    for index in indices:
        filtered_frame.loc[len(filtered_frame)] = testing_frame.loc[index]

    # Add remaining useful life
    filtered_frame['RUL'] = rul_frame['RUL']
    filtered_frame.to_csv('test-regression/processed-data/test.csv', index=False)

    del filtered_frame['RUL']

    # Consider last 30 remaining life cycles as a failure
    label = [0 if x >= 30 else 1 for x in rul_frame['RUL']]
    filtered_frame['BIN'] = label

    filtered_frame.to_csv('test-classification/processed-data/test.csv', index=False)


if __name__ == '__main__':
    # Generate train and test files for FD001 dataset.
    create_train_dataset(file_name='train_FD001.csv')
    create_test_dataset(test_file='test_FD001.csv', rul_file='RUL_FD001.csv')

'''
You can try-out this for FD002, FD003, FD004 datasets.
Use relevant train, test and RUL files.

eg:

createTrainDataset(file_name='train_FD002.csv')
createTestDataset(test_file='test_FD002.csv', rul_file='RUL_FD002.csv')
'''
