'''
A3 Benchmark
------------

This event flow is for generating random forest model for
classification approach
'''

import h2o
from h2o.estimators import H2ORandomForestEstimator

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
model = H2ORandomForestEstimator(ntrees=100, max_depth=20, nbins=100, seed=12345)
model.train(x=input_columns, y=response_column, training_frame=train)

# Test model
performance = model.model_performance(test_data=test)
print performance

'''
Sample Result
-------------

MSE: 0.00438364358677
RMSE: 0.0662090899104
LogLoss: 0.0199020998582
Mean Per-Class Error: 0.0778841712712
AUC: 0.968057488532
Gini: 0.936114977065
Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.155329887122:
       0      1    Error    Rate
-----  -----  ---  -------  ---------------
0      33274  130  0.0039   (130.0/33404.0)
1      94     102  0.4796   (94.0/196.0)
Total  33368  232  0.0067   (224.0/33600.0)
Maximum Metrics: Maximum metrics at their respective thresholds

metric                       threshold    value     idx
---------------------------  -----------  --------  -----
max f1                       0.15533      0.476636  95
max f2                       0.0808625    0.552147  155
max f0point5                 0.231782     0.523743  64
max accuracy                 0.29         0.994881  44
max precision                0.9          1         0
max recall                   6.1507e-05   1         397
max specificity              0.9          1         0
max absolute_mcc             0.121546     0.475736  116
max min_per_class_accuracy   0.0209967    0.918367  272
max mean_per_class_accuracy  0.0217372    0.922116  268
Gains/Lift Table: Avg response rate:  0.58 %

    group    cumulative_data_fraction    lower_threshold    lift       cumulative_lift    response_rate    cumulative_response_rate    capture_rate    cumulative_capture_rate    gain      cumulative_gain
--  -------  --------------------------  -----------------  ---------  -----------------  ---------------  --------------------------  --------------  -------------------------  --------  -----------------
    1        0.01                        0.113745           61.7347    61.7347            0.360119         0.360119                    0.617347        0.617347                   6073.47   6073.47
    2        0.02                        0.068884           14.2857    38.0102            0.0833333        0.221726                    0.142857        0.760204                   1328.57   3701.02
    3        0.030119                    0.05               3.52941    26.4257            0.0205882        0.15415                     0.0357143       0.795918                   252.941   2542.57
    4        0.04                        0.0372535          3.09811    20.6633            0.0180723        0.120536                    0.0306122       0.826531                   209.811   1966.33
    5        0.05                        0.030505           4.08163    17.3469            0.0238095        0.10119                     0.0408163       0.867347                   308.163   1634.69
    6        0.106756                    0.02               1.16863    8.74587            0.00681699       0.0510176                   0.0663265       0.933673                   16.8627   774.587
    7        0.15                        0.0116228          0.589912   6.39456            0.00344116       0.0373016                   0.0255102       0.959184                   -41.0088  539.456
    8        0.2                         0.0101974          0.102041   4.82143            0.000595238      0.028125                    0.00510204      0.964286                   -89.7959  382.143
    9        0.30003                     0.00406962         0.0510052  3.23097            0.00029753       0.0188473                   0.00510204      0.969388                   -94.8995  223.097
    10       0.4                         0.00165463         0.102071   2.44898            0.000595415      0.0142857                   0.0102041       0.979592                   -89.7929  144.898
    11       0.5                         0.000830646        0.0510204  1.96939            0.000297619      0.0114881                   0.00510204      0.984694                   -94.898   96.9388
    12       0.6                         0.000409991        0.0510204  1.64966            0.000297619      0.00962302                  0.00510204      0.989796                   -94.898   64.966
    13       0.70381                     0.000184603        0          1.40634            0                0.00820365                  0               0.989796                   -100      40.6341
    14       0.800089                    4.63605e-05        0.105984   1.24986            0.000618238      0.00729085                  0.0102041       1                          -89.4016  24.9861
    15       1                           0                  0          1                  0                0.00583333                  0               1                          -100      0
'''


