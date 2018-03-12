'''
A3 Benchmark
------------

This event flow is for generating random forest model for
classification approach
'''

import h2o
from h2o.estimators import H2ORandomForestEstimator

print 'A4 Benchmark'
print '------------'

# Initialize H2O server
h2o.init(max_mem_size_GB=5)

# Load train and test data as H2O frames
train = h2o.import_file('processed-data/A4Benchmark_train.csv')
test = h2o.import_file('processed-data/A4Benchmark_test.csv')

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
model = H2ORandomForestEstimator(ntrees=100, max_depth=20, nbins=100, seed=12345)
model.train(x=input_columns, y=response_column, training_frame=train)

# Test model
performance = model.model_performance(test_data=test)
print performance

'''
Sample Result
-------------

MSE: 7.42796967974e-05
RMSE: 0.0086185669805
LogLoss: 0.000561062592878
Mean Per-Class Error: 0.0
AUC: 1.0
Gini: 1.0
Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.281154527828:
       0      1    Error    Rate
-----  -----  ---  -------  -------------
0      33404  0    0        (0.0/33404.0)
1      0      196  0        (0.0/196.0)
Total  33404  196  0        (0.0/33600.0)
Maximum Metrics: Maximum metrics at their respective thresholds

metric                       threshold    value    idx
---------------------------  -----------  -------  -----
max f1                       0.281155     1        44
max f2                       0.281155     1        44
max f0point5                 0.281155     1        44
max accuracy                 0.281155     1        44
max precision                1            1        0
max recall                   0.281155     1        44
max specificity              1            1        0
max absolute_mcc             0.281155     1        44
max min_per_class_accuracy   0.281155     1        44
max mean_per_class_accuracy  0.281155     1        44
Gains/Lift Table: Avg response rate:  0.58 %

    group    cumulative_data_fraction    lower_threshold    lift    cumulative_lift    response_rate    cumulative_response_rate    capture_rate    cumulative_capture_rate    gain    cumulative_gain
--  -------  --------------------------  -----------------  ------  -----------------  ---------------  --------------------------  --------------  -------------------------  ------  -----------------
    1        0.01                        0.00558676         100     100                0.583333         0.583333                    1               1                          9900    9900
    2        0.02                        0.00367006         0       50                 0                0.291667                    0               1                          -100    4900
    3        0.03                        0.00253958         0       33.3333            0                0.194444                    0               1                          -100    3233.33
    4        0.04                        0.00195198         0       25                 0                0.145833                    0               1                          -100    2400
    5        0.05                        0.00160555         0       20                 0                0.116667                    0               1                          -100    1900
    6        0.1                         0.000933795        0       10                 0                0.0583333                   0               1                          -100    900
    7        0.15                        0.000718257        0       6.66667            0                0.0388889                   0               1                          -100    566.667
    8        0.2                         0.000482992        0       5                  0                0.0291667                   0               1                          -100    400
    9        0.3                         0.000116383        0       3.33333            0                0.0194444                   0               1                          -100    233.333
    10       0.4                         2.9248e-05         0       2.5                0                0.0145833                   0               1                          -100    150
    11       0.5                         8.7926e-06         0       2                  0                0.0116667                   0               1                          -100    100
    12       0.6                         2.62507e-06        0       1.66667            0                0.00972222                  0               1                          -100    66.6667
    13       0.7                         6.82732e-07        0       1.42857            0                0.00833333                  0               1                          -100    42.8571
    14       0.8                         1.3114e-07         0       1.25               0                0.00729167                  0               1                          -100    25
    15       0.9                         1.21674e-08        0       1.11111            0                0.00648148                  0               1                          -100    11.1111
    16       1                           1.47779e-69        0       1                  0                0.00583333                  0               1                          -100    0
'''


