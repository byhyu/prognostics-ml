from os.path import join

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


def create_train_dataset(path='test-a1-benchmark/processed-data/', file_name='A1Benchmark_train.csv'):
    training_frame = Frame('csv/' + file_name)
    training_frame = XMLParser.apply_feature_eng(training_frame, 'feature_engineering')

    # Create processed folder for the first time
    if not os.path.exists(join(os.getcwd(), path)):
        os.mkdir(path)

    training_frame.to_csv(path + file_name, index=False)


def create_test_dataset(path='test-a1-benchmark/processed-data/', file_name='A1Benchmark_test.csv'):
    testing_frame = Frame('csv/' + file_name)
    testing_frame = XMLParser.apply_feature_eng(testing_frame, 'feature_engineering')

    # Create processed folder for the first time
    if not os.path.exists(join(os.getcwd(), path)):
        os.mkdir(path)

    testing_frame.to_csv(path + file_name, index=False)


# Create train and test datasets
create_train_dataset(path='test-a1-benchmark\\processed-data\\', file_name='A1Benchmark_train.csv')
create_test_dataset(path='test-a1-benchmark\\processed-data\\', file_name='A1Benchmark_test.csv')

create_train_dataset(path='test-a2-benchmark/processed-data/', file_name='A2Benchmark_train.csv')
create_test_dataset(path='test-a2-benchmark/processed-data/', file_name='A2Benchmark_test.csv')

create_train_dataset(path='test-a3-benchmark/processed-data/', file_name='A3Benchmark_train.csv')
create_test_dataset(path='test-a3-benchmark/processed-data/', file_name='A3Benchmark_test.csv')

create_train_dataset(path='test-a4-benchmark/processed-data/', file_name='A4Benchmark_train.csv')
create_test_dataset(path='test-a4-benchmark/processed-data/', file_name='A4Benchmark_test.csv')
