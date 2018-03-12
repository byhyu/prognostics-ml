from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  mean_squared_error, mean_absolute_error
from sklearn_pandas import DataFrameMapper
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import os

# Loading data
training_frame = pd.read_csv("Training.csv")
testing_frame = pd.read_csv("Testing.csv")

# Feature selection
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

# Parsing data
training_data = training_frame[training_columns]
target_data = training_frame[response_column]
testing_data = testing_frame[training_columns]
ground_truth_data = testing_frame[response_column]

# Setting up mapper
df_mapper = DataFrameMapper([(training_columns, None), (response_column, None)])

# Train data - pandas to sklearn
data = df_mapper.fit_transform(training_frame)
# train
x = data[:, 0:108]
# response
y = data[:, 108]

# Test data - pandas to sklearn
test = df_mapper.fit_transform(testing_frame)
# test
tX = test[:, 0:108]
# ground truth
tY = test[:, 108]

# Setting up algorithm
rf = RandomForestRegressor(max_depth=20)

# Train model
rf.fit(X=x, y=y)

# Get prediction results
result = []
for row in tX:
    if len(row) == 1:
        row = row.reshape(-1, 1)
    elif len(row) > 1:
        row = row.reshape(1, -1)
    result.append(rf.predict(row)[0])

# Analyze performance
print("Performance")
print("-----------")
print("Root Mean Squared Error", mean_squared_error(tY, np.array(result)) ** 0.5)
print("Mean Absolute Error", mean_absolute_error(tY, np.array(result)))

# Dump pickle files
joblib.dump(df_mapper, "mapper.pkl", compress = 3)
joblib.dump(rf, "estimator.pkl", compress = 3)

# Build pmml
command = "java -jar converter-executable-1.1-SNAPSHOT.jar --pkl-mapper-input mapper.pkl --pkl-estimator-input estimator.pkl --pmml-output mapper-estimator.pmml"
os.system(command)

'''
Scikit learn model to pmml
https://github.com/jpmml/sklearn2pmml
https://pypi.python.org/pypi/sklearn-pmml/0.1.0#downloads
'''





