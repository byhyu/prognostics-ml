'''
A3 Benchmark
------------

This event flow is for generating gradient boost model for
classification approach
'''

import h2o
from h2o.estimators import H2OGradientBoostingEstimator

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
model = H2OGradientBoostingEstimator(ntrees=100, max_depth=20, nbins=100, seed=12345)
model.train(x=input_columns, y=response_column, training_frame=train)

# Test model
performance = model.model_performance(test_data=test)
print performance

'''
Sample Result
-------------

MSE: 0.00500776588684
RMSE: 0.0707655699252
LogLoss: 0.0314359291647
Mean Per-Class Error: 0.0788228954616
AUC: 0.966612134316
Gini: 0.933224268632
Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.00150828424997:
       0      1    Error    Rate
-----  -----  ---  -------  ---------------
0      33169  235  0.007    (235.0/33404.0)
1      69     127  0.352    (69.0/196.0)
Total  33238  362  0.009    (304.0/33600.0)
Maximum Metrics: Maximum metrics at their respective thresholds

metric                       threshold    value     idx
---------------------------  -----------  --------  -----
max f1                       0.00150828   0.455197  314
max f2                       0.000580752  0.559157  339
max f0point5                 0.844276     0.49569   52
max accuracy                 0.971565     0.99503   24
max precision                0.999995     1         0
max recall                   1.82084e-06  1         398
max specificity              0.999995     1         0
max absolute_mcc             0.00150828   0.472783  314
max min_per_class_accuracy   2.16976e-05  0.908163  385
max mean_per_class_accuracy  3.41062e-05  0.921177  381
Gains/Lift Table: Avg response rate:  0.58 %

    group    cumulative_data_fraction    lower_threshold    lift       cumulative_lift    response_rate    cumulative_response_rate    capture_rate    cumulative_capture_rate    gain      cumulative_gain
--  -------  --------------------------  -----------------  ---------  -----------------  ---------------  --------------------------  --------------  -------------------------  --------  -----------------
    1        0.01                        0.00215922         59.1837    59.1837            0.345238         0.345238                    0.591837        0.591837                   5818.37   5818.37
    2        0.02                        0.000178209        20.4082    39.7959            0.119048         0.232143                    0.204082        0.795918                   1940.82   3879.59
    3        0.03                        8.10364e-05        3.57143    27.7211            0.0208333        0.161706                    0.0357143       0.831633                   257.143   2672.11
    4        0.04                        5.24112e-05        2.55102    21.4286            0.014881         0.125                       0.0255102       0.857143                   155.102   2042.86
    5        0.05                        3.72097e-05        2.04082    17.551             0.0119048        0.102381                    0.0204082       0.877551                   104.082   1655.1
    6        0.1                         1.59434e-05        0.714286   9.13265            0.00416667       0.0532738                   0.0357143       0.913265                   -28.5714  813.265
    7        0.15                        1.04258e-05        0.408163   6.22449            0.00238095       0.0363095                   0.0204082       0.933673                   -59.1837  522.449
    8        0.2                         7.72935e-06        0.306122   4.7449             0.00178571       0.0276786                   0.0153061       0.94898                    -69.3878  374.49
    9        0.3                         5.13162e-06        0.255102   3.2483             0.0014881        0.0189484                   0.0255102       0.97449                    -74.4898  224.83
    10       0.4                         3.78864e-06        0          2.43622            0                0.0142113                   0               0.97449                    -100      143.622
    11       0.5                         2.91609e-06        0.102041   1.96939            0.000595238      0.0114881                   0.0102041       0.984694                   -89.7959  96.9388
    12       0.6                         2.3091e-06         0          1.64116            0                0.00957341                  0               0.984694                   -100      64.1156
    13       0.7                         1.8348e-06         0.0510204  1.41399            0.000297619      0.0082483                   0.00510204      0.989796                   -94.898   41.3994
    14       0.8                         1.41505e-06        0.102041   1.25               0.000595238      0.00729167                  0.0102041       1                          -89.7959  25
    15       0.9                         1.01798e-06        0          1.11111            0                0.00648148                  0               1                          -100      11.1111
    16       1                           2.5878e-09         0          1                  0                0.00583333                  0               1                          -100      0
'''


