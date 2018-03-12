'''
This event flow is for generating random forest model for
classification approach
'''

import h2o
from h2o.estimators import H2ORandomForestEstimator

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
model = H2ORandomForestEstimator(ntrees=100, max_depth=20, nbins=100, seed=12345)
model.train(x=input_columns, y=response_column, training_frame=train)

# Test model
performance = model.model_performance(test_data=test)
print performance

'''
Sample Result
-------------

MSE: 0.0635067962679
RMSE: 0.25200554809
LogLoss: 0.195672790572
Mean Per-Class Error: 0.0533333333333
AUC: 0.980266666667
Gini: 0.960533333333
Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.1:
       0    1    Error    Rate
-----  ---  ---  -------  -----------
0      70   5    0.0667   (5.0/75.0)
1      1    24   0.04     (1.0/25.0)
Total  71   29   0.06     (6.0/100.0)
Maximum Metrics: Maximum metrics at their respective thresholds

metric                       threshold    value     idx
---------------------------  -----------  --------  -----
max f1                       0.1          0.888889  23
max f2                       0.1          0.930233  23
max f0point5                 0.4          0.904762  14
max accuracy                 0.1          0.94      23
max precision                1            1         0
max recall                   0.03         1         31
max specificity              1            1         0
max absolute_mcc             0.1          0.852484  23
max min_per_class_accuracy   0.1          0.933333  23
max mean_per_class_accuracy  0.1          0.946667  23
Gains/Lift Table: Avg response rate: 25.00 %

    group    cumulative_data_fraction    lower_threshold    lift      cumulative_lift    response_rate    cumulative_response_rate    capture_rate    cumulative_capture_rate    gain      cumulative_gain
--  -------  --------------------------  -----------------  --------  -----------------  ---------------  --------------------------  --------------  -------------------------  --------  -----------------
    1        0.04                        1                  4         4                  1                1                           0.16            0.16                       300       300
    2        0.04                        0.9904             0         4                  0                1                           0               0.16                       -100      300
    3        0.06                        0.99               4         4                  1                1                           0.08            0.24                       300       300
    4        0.11                        0.88               4         4                  1                1                           0.2             0.44                       300       300
    5        0.15                        0.686              4         4                  1                1                           0.16            0.6                        300       300
    6        0.2                         0.376              3.2       3.8                0.8              0.95                        0.16            0.76                       220       280
    7        0.31                        0.09               1.81818   3.09677            0.454545         0.774194                    0.2             0.96                       81.8182   209.677
    8        0.42                        0.01               0.363636  2.38095            0.0909091        0.595238                    0.04            1                          -63.6364  138.095
    9        1                           0                  0         1                  0                0.25                        0               1                          -100      0
'''


