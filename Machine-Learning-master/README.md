# Feature Engineering for Machine Learning
This repository will give you a basic idea about how to 
use feature engineering along side with machine learning.
All the new feature engineering processes that describe here
have been carried out using python featureeng package.


#### Sample Datasets
1. NASA   : 
[Turbofan engine degradataion](https://ti.arc.nasa.gov/c/6/)

2. Yahoo  : 
[Webscope](https://webscope.sandbox.yahoo.com/)

Download datasets from above links. Copy extracted files into 
*dataset* folder in relevant project.

#### Setup Environment
All the dependencies given below need to be installed using 
*python pip install* in python 2.7

##### 1. Numpy
```
# pip install numpy
```
##### 2. Scipy
```
# pip install scipy
```
##### 3. Pandas
```
# pip install pandas
```
##### 4. H20
```
# pip install requests
# pip install tabulate
# pip install scikit-learn
# pip uninstall h2o
# pip install http://h2o-release.s3.amazonaws.com/h2o/rel-tutte/1/Python/h2o-3.10.2.1-py2.py3-none-any.whl
```
##### 5. Sklearn-Pandas
```
# pip install sklearn-pandas
```

##### NASA - Turbofan Engine Degradation

After copying data files into *dataset* folder executes following
python scripts in order.

###### 01-ConvertToCSV.py
This will convert all the data files in .txt 

###### 02-ProcessDataset.py
This script will create data frames (training, testing) with engineered features for regression and classification 
approach. Feature engineering configurations will be loaded from 'feature_engineering.xml' file.

```
<flow>
    <moving_k_closest_average window="5" k_closest="3">
        <feature>Sensor1</feature>
        <feature>Sensor2</feature>
        <feature>Sensor3</feature>
        <feature>Sensor4</feature>
        <feature>Sensor5</feature>
        <feature>Sensor6</feature>
        <feature>Sensor7</feature>
        <feature>Sensor9</feature>
        <feature>Sensor10</feature>
        <feature>Sensor11</feature>
        <feature>Sensor12</feature>
        <feature>Sensor13</feature>
        <feature>Sensor14</feature>
        <feature>Sensor15</feature>
        <feature>Sensor16</feature>
        <feature>Sensor17</feature>
        <feature>Sensor18</feature>
        <feature>Sensor19</feature>
        <feature>Sensor20</feature>
        <feature>Sensor21</feature>
    </moving_k_closest_average>
    <moving_variance window="5">
        <feature>Sensor1</feature>
        <feature>Sensor2</feature>
        <feature>Sensor3</feature>
        <feature>Sensor4</feature>
        <feature>Sensor5</feature>
        <feature>Sensor6</feature>
        <feature>Sensor7</feature>
        <feature>Sensor9</feature>
        <feature>Sensor10</feature>
        <feature>Sensor11</feature>
        <feature>Sensor12</feature>
        <feature>Sensor13</feature>
        <feature>Sensor14</feature>
        <feature>Sensor15</feature>
        <feature>Sensor16</feature>
        <feature>Sensor17</feature>
        <feature>Sensor18</feature>
        <feature>Sensor19</feature>
        <feature>Sensor20</feature>
        <feature>Sensor21</feature>
    </moving_variance>
</flow>
```

Processed csv files will be stored in the 'processed-data' directory
in each test. So data processing happens only once which allows you to 
compare different algorithms with similar features.

There are two directories, **test-regression** and **test-classification** 
separately. 
 
Classification test has been carried out using three algorithms. 

* Deep Learning
* Random Forest
* Gradient Boost

Regression test has been carried out using four algorithms.

* Deep Learning
* Random Forest
* Gradient Boost
* Generalized Linear

You can simply run each script and test the performance for
each model

##### Yahoo - Web Scope
*The Yahoo Webscope Program is a reference library of interesting and scientifically useful datasets for non-commercial 
use by academics and other scientists.*

###### 01-MergeFiles.py
This will merge individual benchmark data files and split them to 
train and test frames 

###### 02-ProcessDataset.py
This script will create data frames (training, testing) with engineered features for classification 
approach. Feature engineering configurations will be loaded from 'feature_engineering.xml' file.

There are four directories for each benchmark test. Each benchmark test
has been carried out using three machine learning algorithms. Run each 
script to check the model performance in each approach.