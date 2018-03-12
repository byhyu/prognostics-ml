'''
This event flow is for generating generalized linear model for
regression approach
'''

import h2o
from h2o.estimators import H2OGeneralizedLinearEstimator

# Initialize H2O server
h2o.init(max_mem_size_GB=5)

# Load train and test data as H2O frames
train = h2o.import_file('processed-data/train.csv')
test = h2o.import_file('processed-data/test.csv')

# Define input and response columns
response_column = 'RUL'
input_columns = train.col_names
input_columns.remove('UnitNumber')
input_columns.remove('Time')
input_columns.remove('Setting1')
input_columns.remove('Setting2')
input_columns.remove('Setting3')
input_columns.remove('RUL')

# Define model and train model
model = H2OGeneralizedLinearEstimator(nfolds=10)
model.train(x=input_columns, y=response_column, training_frame=train)

# Test model
performance = model.model_performance(test_data=test)
print performance

'''
Sample Result
-------------

MSE: 968.727782044
RMSE: 31.1243920751
MAE: 25.0215352607
RMSLE: NaN
R^2: 0.4390266746
Mean Residual Deviance: 968.727782044
Null degrees of freedom: 99
Residual degrees of freedom: 61
Null deviance: 276937.562965
Residual deviance: 96872.7782044
AIC: 1051.38607121
'''


