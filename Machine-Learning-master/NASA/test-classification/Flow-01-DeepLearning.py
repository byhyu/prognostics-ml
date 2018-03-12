'''
This event flow is for generating deep learning model for
classification approach
'''

import h2o
from h2o.estimators import H2ODeepLearningEstimator

# Initialize H2O server
h2o.init(max_mem_size_GB=5)

# Load train and test data as H2O frames
train = h2o.import_file('processed-data/train.csv')
test = h2o.import_file('processed-data/test.csv')

# Define input and response columns
response_column = 'BIN'
input_columns = train.col_names
input_columns.remove('UnitNumber')
input_columns.remove('Time')
input_columns.remove('Setting1')
input_columns.remove('Setting2')
input_columns.remove('Setting3')
input_columns.remove('BIN')

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

MSE: 0.0505274258023
RMSE: 0.224783063869
LogLoss: 0.196900085695
Mean Per-Class Error: 0.0466666666667
AUC: 0.984
Gini: 0.968
Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.336405428114:
       0    1    Error    Rate
-----  ---  ---  -------  -----------
0      72   3    0.04     (3.0/75.0)
1      2    23   0.08     (2.0/25.0)
Total  74   26   0.05     (5.0/100.0)
Maximum Metrics: Maximum metrics at their respective thresholds

metric                       threshold    value     idx
---------------------------  -----------  --------  -----
max f1                       0.336405     0.901961  25
max f2                       0.0118001    0.94697   31
max f0point5                 0.477617     0.909091  23
max accuracy                 0.477617     0.95      23
max precision                0.999999     1         0
max recall                   0.0118001    1         31
max specificity              0.999999     1         0
max absolute_mcc             0.336405     0.868722  25
max min_per_class_accuracy   0.336405     0.92      25
max mean_per_class_accuracy  0.0118001    0.953333  31
Gains/Lift Table: Avg response rate: 25.00 %

    group    cumulative_data_fraction    lower_threshold    lift    cumulative_lift    response_rate    cumulative_response_rate    capture_rate    cumulative_capture_rate    gain    cumulative_gain
--  -------  --------------------------  -----------------  ------  -----------------  ---------------  --------------------------  --------------  -------------------------  ------  -----------------
    1        0.01                        0.999995           4       4                  1                1                           0.04            0.04                       300     300
    2        0.02                        0.999972           4       4                  1                1                           0.04            0.08                       300     300
    3        0.03                        0.999818           4       4                  1                1                           0.04            0.12                       300     300
    4        0.04                        0.999679           4       4                  1                1                           0.04            0.16                       300     300
    5        0.05                        0.999516           4       4                  1                1                           0.04            0.2                        300     300
    6        0.1                         0.998849           4       4                  1                1                           0.2             0.4                        300     300
    7        0.15                        0.96929            3.2     3.73333            0.8              0.933333                    0.16            0.56                       220     273.333
    8        0.2                         0.828816           3.2     3.6                0.8              0.9                         0.16            0.72                       220     260
    9        0.3                         0.0208702          2.4     3.2                0.6              0.8                         0.24            0.96                       140     220
    10       0.4                         0.000134588        0.4     2.5                0.1              0.625                       0.04            1                          -60     150
    11       0.5                         1.34229e-12        0       2                  0                0.5                         0               1                          -100    100
    12       0.6                         6.16121e-17        0       1.66667            0                0.416667                    0               1                          -100    66.6667
    13       0.7                         2.17058e-22        0       1.42857            0                0.357143                    0               1                          -100    42.8571
    14       0.8                         7.03117e-31        0       1.25               0                0.3125                      0               1                          -100    25
    15       0.9                         2.95017e-41        0       1.11111            0                0.277778                    0               1                          -100    11.1111
    16       1                           1.03898e-73        0       1                  0                0.25                        0               1                          -100    0
'''


