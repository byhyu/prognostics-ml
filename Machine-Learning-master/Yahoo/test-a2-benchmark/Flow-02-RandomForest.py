'''
A1 Benchmark
------------

This event flow is for generating random forest model for
classification approach
'''

import h2o
from h2o.estimators import H2ORandomForestEstimator

print 'A2 Benchmark'
print '------------'

# Initialize H2O server
h2o.init(max_mem_size_GB=5)

# Load train and test data as H2O frames
train = h2o.import_file('processed-data/A2Benchmark_train.csv')
test = h2o.import_file('processed-data/A2Benchmark_test.csv')

# Define input and response columns
response_column = 'is_anomaly'
input_columns = train.col_names
input_columns.remove(response_column)
input_columns.remove('timestamp')

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

MSE: 0.00391366356838
RMSE: 0.0625592804337
LogLoss: 0.0113306658575
Mean Per-Class Error: 0.00940566104327
AUC: 0.999066369437
Gini: 0.998132738874
Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.64:
       0      1    Error    Rate
-----  -----  ---  -------  -------------
0      28331  3    0.0001   (3.0/28334.0)
1      5      81   0.0581   (5.0/86.0)
Total  28336  84   0.0003   (8.0/28420.0)
Maximum Metrics: Maximum metrics at their respective thresholds

metric                       threshold    value     idx
---------------------------  -----------  --------  -----
max f1                       0.64         0.952941  17
max f2                       0.64         0.946262  17
max f0point5                 0.71         0.972906  15
max accuracy                 0.71         0.999719  15
max precision                1            1         0
max recall                   0.03         1         68
max specificity              1            1         0
max absolute_mcc             0.64         0.952866  17
max min_per_class_accuracy   0.38         0.984471  31
max mean_per_class_accuracy  0.03         0.990594  68
Gains/Lift Table: Avg response rate:  0.30 %

    group    cumulative_data_fraction    lower_threshold    lift       cumulative_lift    response_rate    cumulative_response_rate    capture_rate    cumulative_capture_rate    gain      cumulative_gain
--  -------  --------------------------  -----------------  ---------  -----------------  ---------------  --------------------------  --------------  -------------------------  --------  -----------------
    1        0.0181562                   0.49               51.8753    51.8753            0.156977         0.156977                    0.94186         0.94186                    5087.53   5087.53
    2        0.0200563                   0.08               24.4789    49.2799            0.0740741        0.149123                    0.0465116       0.988372                   2347.89   4827.99
    3        1                           0                  0.0118659  1                  3.59066e-05      0.00302604                  0.0116279       1                          -98.8134  0
'''


