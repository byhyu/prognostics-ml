'''
A1 Benchmark
------------

This event flow is for generating gradient boost model for
classification approach
'''

import h2o
from h2o.estimators import H2OGradientBoostingEstimator

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
model = H2OGradientBoostingEstimator(ntrees=100, max_depth=20, nbins=100, seed=12345)
model.train(x=input_columns, y=response_column, training_frame=train)

# Test model
performance = model.model_performance(test_data=test)
print performance

'''
Sample Result
-------------

MSE: 0.000723878601322
RMSE: 0.0269049921264
LogLoss: 0.016027820498
Mean Per-Class Error: 0.00810801715746
AUC: 0.988189675975
Gini: 0.97637935195
Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.583571568698:
       0      1    Error    Rate
-----  -----  ---  -------  --------------
0      28317  17   0.0006   (17.0/28334.0)
1      2      84   0.0233   (2.0/86.0)
Total  28319  101  0.0007   (19.0/28420.0)
Maximum Metrics: Maximum metrics at their respective thresholds

metric                       threshold    value     idx
---------------------------  -----------  --------  -----
max f1                       0.583572     0.898396  7
max f2                       0.583572     0.94382   7
max f0point5                 1            0.890411  1
max accuracy                 1            0.999367  1
max precision                1            0.886364  1
max recall                   3.16466e-07  1         395
max specificity              1            0.999647  0
max absolute_mcc             0.583572     0.900985  7
max min_per_class_accuracy   7.09508e-05  0.988372  114
max mean_per_class_accuracy  7.09508e-05  0.991892  114
Gains/Lift Table: Avg response rate:  0.30 %

    group    cumulative_data_fraction    lower_threshold    lift      cumulative_lift    response_rate    cumulative_response_rate    capture_rate    cumulative_capture_rate    gain      cumulative_gain
--  -------  --------------------------  -----------------  --------  -----------------  ---------------  --------------------------  --------------  -------------------------  --------  -----------------
    1        0.0100281                   3.03908e-05        98.5598   98.5598            0.298246         0.298246                    0.988372        0.988372                   9755.98   9755.98
    2        0.0200211                   1.15688e-05        0         49.3665            0                0.149385                    0               0.988372                   -100      4836.65
    3        0.0300141                   7.28088e-06        0         32.9303            0                0.0996483                   0               0.988372                   -100      3193.03
    4        0.040007                    5.28162e-06        0         24.705             0                0.0747581                   0               0.988372                   -100      2370.5
    5        0.05                        4.55913e-06        0         19.7674            0                0.059817                    0               0.988372                   -100      1876.74
    6        0.100035                    3.65234e-06        0         9.88024            0                0.029898                    0               0.988372                   -100      888.024
    7        0.213406                    2.44016e-06        0         4.63142            0                0.0140148                   0               0.988372                   -100      363.142
    8        0.377762                    2.05417e-06        0         2.61639            0                0.00791729                  0               0.988372                   -100      161.639
    9        0.815623                    2.003e-06          0         1.2118             0                0.00366695                  0               0.988372                   -100      21.18
    10       0.967734                    1.94215e-06        0         1.02133            0                0.00309057                  0               0.988372                   -100      2.13262
    11       1                           1e-19              0.360376  1                  0.00109051       0.00302604                  0.0116279       1                          -63.9624  0
'''


