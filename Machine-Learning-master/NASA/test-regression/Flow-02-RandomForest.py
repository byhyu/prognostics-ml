'''
This event flow is for generating random forest model for
regression approach
'''

import h2o
from h2o.estimators import H2ORandomForestEstimator

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
model = H2ORandomForestEstimator(ntrees=100, max_depth=20, nbins=100, seed=12345)
model.train(x=input_columns, y=response_column, training_frame=train)

# Test model
performance = model.model_performance(test_data=test)
print performance

'''
Sample Result
-------------

MSE: 1050.54544798
RMSE: 32.4121188443
MAE: 22.8847264328
RMSLE: 0.363088934639
Mean Residual Deviance: 1050.54544798
'''


