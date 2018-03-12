from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np

print("Loading Data")
training_frame = pd.read_csv("Training.csv")
testing_frame = pd.read_csv("Testing.csv")

print("Selecting Required Columns")
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

print("Parsing Data")
training_data = training_frame[training_columns]
target_data = training_frame[response_column]
testing_data = testing_frame[training_columns]
ground_truth_data = testing_frame[response_column]

print("Setting Up Algorithm")
rf = RandomForestRegressor(max_depth=20)

print("Train Model")
rf.fit(X=training_data.get_values(), y=target_data.get_values())

print("Predict and Collect Result")
result = []
for col in testing_data.get_values():
    if len(col) == 1:
        col = col.reshape(-1, 1)
    elif len(col) > 1:
        col = col.reshape(1, -1)
    result.append(rf.predict(col.reshape(1, -1))[0])

print("Performance")
print("Root Mean Squared Error", mean_squared_error(ground_truth_data, np.array(result)) ** 0.5)
print("Mean Absolute Error", mean_absolute_error(ground_truth_data, np.array(result)))

'''
Scikit learn model to pmml
https://github.com/jpmml/sklearn2pmml
https://pypi.python.org/pypi/sklearn-pmml/0.1.0#downloads
'''



