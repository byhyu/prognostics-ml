'''
A3 Benchmark
------------

This event flow is for generating deep learning model for
classification approach
'''

import h2o
from h2o.estimators import H2ODeepLearningEstimator

print 'A3 Benchmark'
print '------------'

# Initialize H2O server
h2o.init(max_mem_size_GB=5)

# Load train and test data as H2O frames
train = h2o.import_file('processed-data/A3Benchmark_train.csv')
test = h2o.import_file('processed-data/A3Benchmark_test.csv')

# Define input and response columns
response_column = 'anomaly'
input_columns = train.col_names
input_columns.remove(response_column)
input_columns.remove('timestamps')

print 'Input columns   :', input_columns
print 'Response column :', response_column

# Explicitly imply response column contains label data
train[response_column] = train[response_column].asfactor()
test[response_column] = test[response_column].asfactor()

# Define model and train model
model = H2ODeepLearningEstimator(hidden=[500, 500], nfolds=10, epochs=100)
model.train(x=input_columns, y=response_column, training_frame=train)

# Test model
performance = model.model_performance(test_data=test)
print performance

'''
Sample Result
-------------

MSE: 0.000344255969462
RMSE: 0.018554136182
LogLoss: 0.00133353905738
Mean Per-Class Error: 7.48413363669e-05
AUC: 0.999999236313
Gini: 0.999998472626
Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.0136150933157:
       0      1    Error    Rate
-----  -----  ---  -------  -------------
0      33404  0    0        (0.0/33404.0)
1      1      195  0.0051   (1.0/196.0)
Total  33405  195  0        (1.0/33600.0)
Maximum Metrics: Maximum metrics at their respective thresholds

metric                       threshold    value     idx
---------------------------  -----------  --------  -----
max f1                       0.0136151    0.997442  58
max f2                       0.0136151    0.995914  58
max f0point5                 0.0136151    0.998975  58
max accuracy                 0.0136151    0.99997   58
max precision                1            1         0
max recall                   0.00854961   1         64
max specificity              1            1         0
max absolute_mcc             0.0136151    0.997431  58
max min_per_class_accuracy   0.00854961   0.99985   64
max mean_per_class_accuracy  0.00854961   0.999925  64
Gains/Lift Table: Avg response rate:  0.58 %

    group    cumulative_data_fraction    lower_threshold    lift    cumulative_lift    response_rate    cumulative_response_rate    capture_rate    cumulative_capture_rate    gain    cumulative_gain
--  -------  --------------------------  -----------------  ------  -----------------  ---------------  --------------------------  --------------  -------------------------  ------  -----------------
    1        0.01                        0.0014351          100     100                0.583333         0.583333                    1               1                          9900    9900
    2        0.02                        0.000585087        0       50                 0                0.291667                    0               1                          -100    4900
    3        0.03                        0.000430054        0       33.3333            0                0.194444                    0               1                          -100    3233.33
    4        0.04                        0.000355075        0       25                 0                0.145833                    0               1                          -100    2400
    5        0.05                        0.00030352         0       20                 0                0.116667                    0               1                          -100    1900
    6        0.1                         0.000192231        0       10                 0                0.0583333                   0               1                          -100    900
    7        0.15                        0.000127954        0       6.66667            0                0.0388889                   0               1                          -100    566.667
    8        0.2                         7.88971e-05        0       5                  0                0.0291667                   0               1                          -100    400
    9        0.3                         1.75277e-05        0       3.33333            0                0.0194444                   0               1                          -100    233.333
    10       0.4                         2.32549e-06        0       2.5                0                0.0145833                   0               1                          -100    150
    11       0.5                         3.55444e-07        0       2                  0                0.0116667                   0               1                          -100    100
    12       0.6                         5.79054e-08        0       1.66667            0                0.00972222                  0               1                          -100    66.6667
    13       0.7                         6.98278e-09        0       1.42857            0                0.00833333                  0               1                          -100    42.8571
    14       0.8                         5.32814e-10        0       1.25               0                0.00729167                  0               1                          -100    25
    15       0.9                         1.50098e-11        0       1.11111            0                0.00648148                  0               1                          -100    11.1111
    16       1                           7.50488e-30        0       1                  0                0.00583333                  0               1                          -100    0
'''


