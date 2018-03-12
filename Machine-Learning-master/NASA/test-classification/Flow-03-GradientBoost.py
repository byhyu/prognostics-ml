'''
This event flow is for generating gradient boosting model for
classification approach
'''

import h2o
from h2o.estimators import H2OGradientBoostingEstimator

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
model = H2OGradientBoostingEstimator(ntrees=100, max_depth=20, nbins=100, seed=12345)
model.train(x=input_columns, y=response_column, training_frame=train)

# Test model
performance = model.model_performance(test_data=test)
print performance

'''
Sample Result
-------------

MSE: 0.0827928827336
RMSE: 0.287737524028
LogLoss: 0.387905551816
Mean Per-Class Error: 0.06
AUC: 0.979733333333
Gini: 0.959466666667
Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.00183417929204:
       0    1    Error    Rate
-----  ---  ---  -------  -----------
0      68   7    0.0933   (7.0/75.0)
1      1    24   0.04     (1.0/25.0)
Total  69   31   0.08     (8.0/100.0)
Maximum Metrics: Maximum metrics at their respective thresholds

metric                       threshold    value     idx
---------------------------  -----------  --------  -----
max f1                       0.00183418   0.857143  30
max f2                       0.000665949  0.932836  33
max f0point5                 0.965977     0.898876  15
max accuracy                 0.0296534    0.93      21
max precision                0.999945     1         0
max recall                   0.000665949  1         33
max specificity              0.999945     1         0
max absolute_mcc             0.00183418   0.811423  30
max min_per_class_accuracy   0.00446925   0.92      28
max mean_per_class_accuracy  0.000665949  0.94      33
Gains/Lift Table: Avg response rate: 25.00 %

    group    cumulative_data_fraction    lower_threshold    lift    cumulative_lift    response_rate    cumulative_response_rate    capture_rate    cumulative_capture_rate    gain    cumulative_gain
--  -------  --------------------------  -----------------  ------  -----------------  ---------------  --------------------------  --------------  -------------------------  ------  -----------------
    1        0.01                        0.999908           4       4                  1                1                           0.04            0.04                       300     300
    2        0.02                        0.999901           4       4                  1                1                           0.04            0.08                       300     300
    3        0.03                        0.999889           4       4                  1                1                           0.04            0.12                       300     300
    4        0.04                        0.999861           4       4                  1                1                           0.04            0.16                       300     300
    5        0.05                        0.999856           4       4                  1                1                           0.04            0.2                        300     300
    6        0.1                         0.999335           4       4                  1                1                           0.2             0.4                        300     300
    7        0.15                        0.966895           4       4                  1                1                           0.2             0.6                        300     300
    8        0.2                         0.16774            2.4     3.6                0.6              0.9                         0.12            0.72                       140     260
    9        0.3                         0.00192386         2       3.06667            0.5              0.766667                    0.2             0.92                       100     206.667
    10       0.4                         8.1796e-05         0.8     2.5                0.2              0.625                       0.08            1                          -20     150
    11       0.5                         3.54999e-05        0       2                  0                0.5                         0               1                          -100    100
    12       0.6                         2.70499e-05        0       1.66667            0                0.416667                    0               1                          -100    66.6667
    13       0.7                         2.50172e-05        0       1.42857            0                0.357143                    0               1                          -100    42.8571
    14       0.83                        2.36545e-05        0       1.20482            0                0.301205                    0               1                          -100    20.4819
    15       0.9                         2.36231e-05        0       1.11111            0                0.277778                    0               1                          -100    11.1111
    16       1                           1.8649e-05         0       1                  0                0.25                        0               1                          -100    0
'''


