'''
This event flow is for generating deep learning model for
regression approach
'''

import h2o
from h2o.estimators import H2ODeepLearningEstimator

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
model = H2ODeepLearningEstimator(hidden=[500, 500], nfolds=10, epochs=100)
model.train(x=input_columns, y=response_column, training_frame=train)

# Test model
performance = model.model_performance(test_data=test)
print performance

'''
Sample Result
-------------

MSE: 1034.3309604
RMSE: 32.1610161594
MAE: 23.5682534367
RMSLE: NaN
Mean Residual Deviance: 1034.3309604
'''


