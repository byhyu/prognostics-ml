'''
A3 Benchmark
------------

This event flow is for generating deep learning model for
classification approach
'''

import h2o
from h2o.estimators import H2ODeepLearningEstimator

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
model = H2ODeepLearningEstimator(hidden=[200, 200], nfolds=10, epochs=100)
model.train(x=input_columns, y=response_column, training_frame=train)

# Test model
performance = model.model_performance(test_data=test)
print performance

'''
Sample Result
-------------

MSE: 0.00461413442643
RMSE: 0.0679274202839
LogLoss: 0.0354236501236
Mean Per-Class Error: 0.106653180971
AUC: 0.940574986131
Gini: 0.881149972263
Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.00223926332747:
       0      1    Error    Rate
-----  -----  ---  -------  ---------------
0      33272  132  0.004    (132.0/33404.0)
1      92     104  0.4694   (92.0/196.0)
Total  33364  236  0.0067   (224.0/33600.0)
Maximum Metrics: Maximum metrics at their respective thresholds

metric                       threshold    value     idx
---------------------------  -----------  --------  -----
max f1                       0.00223926   0.481481  213
max f2                       0.00172859   0.515414  225
max f0point5                 0.867526     0.552632  29
max accuracy                 0.917207     0.995298  27
max precision                0.999999     1         0
max recall                   3.77176e-07  1         399
max specificity              0.999999     1         0
max absolute_mcc             0.00223926   0.480245  213
max min_per_class_accuracy   1.01941e-05  0.887755  381
max mean_per_class_accuracy  1.21136e-05  0.893347  379
Gains/Lift Table: Avg response rate:  0.58 %

    group    cumulative_data_fraction    lower_threshold    lift       cumulative_lift    response_rate    cumulative_response_rate    capture_rate    cumulative_capture_rate    gain      cumulative_gain
--  -------  --------------------------  -----------------  ---------  -----------------  ---------------  --------------------------  --------------  -------------------------  --------  -----------------
    1        0.01                        0.001087           55.6122    55.6122            0.324405         0.324405                    0.556122        0.556122                   5461.22   5461.22
    2        0.02                        0.000254926        9.69388    32.6531            0.0565476        0.190476                    0.0969388       0.653061                   869.388   3165.31
    3        0.03                        9.00204e-05        7.65306    24.3197            0.0446429        0.141865                    0.0765306       0.729592                   665.306   2331.97
    4        0.04                        4.40015e-05        3.57143    19.1327            0.0208333        0.111607                    0.0357143       0.765306                   257.143   1813.27
    5        0.05                        2.81509e-05        4.08163    16.1224            0.0238095        0.0940476                   0.0408163       0.806122                   308.163   1512.24
    6        0.1                         1.11226e-05        1.42857    8.77551            0.00833333       0.0511905                   0.0714286       0.877551                   42.8571   777.551
    7        0.15                        7.36188e-06        0.408163   5.98639            0.00238095       0.0349206                   0.0204082       0.897959                   -59.1837  498.639
    8        0.2                         5.53771e-06        0.204082   4.54082            0.00119048       0.0264881                   0.0102041       0.908163                   -79.5918  354.082
    9        0.3                         3.81165e-06        0.306122   3.12925            0.00178571       0.018254                    0.0306122       0.938776                   -69.3878  212.925
    10       0.4                         2.93411e-06        0.102041   2.37245            0.000595238      0.0138393                   0.0102041       0.94898                    -89.7959  137.245
    11       0.5                         2.37096e-06        0.0510204  1.90816            0.000297619      0.011131                    0.00510204      0.954082                   -94.898   90.8163
    12       0.6                         1.9436e-06         0.204082   1.62415            0.00119048       0.00947421                  0.0204082       0.97449                    -79.5918  62.415
    13       0.7                         1.60347e-06        0.102041   1.40671            0.000595238      0.00820578                  0.0102041       0.984694                   -89.7959  40.6706
    14       0.8                         1.31861e-06        0.0510204  1.23724            0.000297619      0.00721726                  0.00510204      0.989796                   -94.898   23.7245
    15       0.9                         1.03855e-06        0          1.09977            0                0.00641534                  0               0.989796                   -100      9.97732
    16       1                           1.80252e-10        0.102041   1                  0.000595238      0.00583333                  0.0102041       1                          -89.7959  0
'''


