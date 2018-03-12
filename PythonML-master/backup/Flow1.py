from dataprocessor import TestData, TrainData
import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import time

# Connect to the local H2O server
h2o.init(ip="127.0.0.1", port="54321")
# In case cluster was already running
h2o.remove_all()

start = time.clock()
print("Flow initialized")
print("----------------")
training_frame = TrainData.process()
testing_frame = TestData.process()

# Creating h2o frames
train = h2o.H2OFrame(training_frame)
test = h2o.H2OFrame(testing_frame)

# Applying haders
train.set_names(list(training_frame.columns))
test.set_names(list(testing_frame.columns))

training_columns = ['Setting1', 'Setting2', 'Setting3', 'Sensor1', 'Sensor2', 'Sensor3', 'Sensor4', 'Sensor5',
                    'Sensor6', 'Sensor7', 'Sensor8', 'Sensor9', 'Sensor10', 'Sensor11', 'Sensor12', 'Sensor13',
                    'Sensor14', 'Sensor15', 'Sensor16', 'Sensor17', 'Sensor18', 'Sensor19', 'Sensor20',
                    'Sensor21', 'Sensor1_ma_5', 'Sensor2_ma_5', 'Sensor3_ma_5', 'Sensor4_ma_5', 'Sensor5_ma_5',
                    'Sensor6_ma_5', 'Sensor7_ma_5', 'Sensor8_ma_5', 'Sensor9_ma_5', 'Sensor10_ma_5',
                    'Sensor11_ma_5', 'Sensor12_ma_5', 'Sensor13_ma_5', 'Sensor14_ma_5', 'Sensor15_ma_5',
                    'Sensor16_ma_5', 'Sensor17_ma_5', 'Sensor18_ma_5', 'Sensor19_ma_5', 'Sensor20_ma_5',
                    'Sensor21_ma_5', 'Sensor1_mm_5', 'Sensor2_mm_5', 'Sensor3_mm_5', 'Sensor4_mm_5',
                    'Sensor5_mm_5', 'Sensor6_mm_5', 'Sensor7_mm_5', 'Sensor8_mm_5', 'Sensor9_mm_5',
                    'Sensor10_mm_5', 'Sensor11_mm_5', 'Sensor12_mm_5', 'Sensor13_mm_5', 'Sensor14_mm_5',
                    'Sensor15_mm_5', 'Sensor16_mm_5', 'Sensor17_mm_5', 'Sensor18_mm_5', 'Sensor19_mm_5',
                    'Sensor20_mm_5', 'Sensor21_mm_5', 'Sensor1_sd_10', 'Sensor2_sd_10', 'Sensor3_sd_10',
                    'Sensor4_sd_10', 'Sensor5_sd_10', 'Sensor6_sd_10', 'Sensor7_sd_10', 'Sensor8_sd_10',
                    'Sensor9_sd_10', 'Sensor10_sd_10', 'Sensor11_sd_10', 'Sensor12_sd_10', 'Sensor13_sd_10',
                    'Sensor14_sd_10', 'Sensor15_sd_10', 'Sensor16_sd_10', 'Sensor17_sd_10', 'Sensor18_sd_10',
                    'Sensor19_sd_10', 'Sensor20_sd_10', 'Sensor21_sd_10', 'Sensor1_entropy_250',
                    'Sensor2_entropy_250', 'Sensor3_entropy_250', 'Sensor4_entropy_250', 'Sensor5_entropy_250',
                    'Sensor6_entropy_250', 'Sensor7_entropy_250', 'Sensor8_entropy_250', 'Sensor9_entropy_250',
                    'Sensor10_entropy_250', 'Sensor11_entropy_250', 'Sensor12_entropy_250', 'Sensor13_entropy_250',
                    'Sensor14_entropy_250', 'Sensor15_entropy_250', 'Sensor16_entropy_250', 'Sensor17_entropy_250',
                    'Sensor18_entropy_250', 'Sensor19_entropy_250', 'Sensor20_entropy_250', 'Sensor21_entropy_250']

response_column = 'RUL'

# Building model
model = H2ODeepLearningEstimator(hidden=[1000, 1000, 1000], score_each_iteration=True, variable_importances=True)
model.show()

# Training model
model.train(x=training_columns, y=response_column, training_frame=train)

# Performance testing
performance = model.model_performance(test_data=test)
print("\nPerformance data")
print("----------------------------------------------------------------------------------------------------------------")
performance.show()

print("\nTime taken : ", time.clock() - start)

