# Split files
import pandas as pd
from os import listdir
from os.path import isfile, join

'''
Merge Files
-----------

Merge files in particular benchmark and output train
and test csv files

Input directory  : dataset
Output directory : csv
'''
def merge(directory='A1Benchmark'):
    path = 'dataset/' + directory + '/'
    csv_files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.csv')]

    train = pd.DataFrame()
    test = pd.DataFrame()

    # Append files up to last 20, to train frame
    # You can change the ratios as you want
    for file in csv_files[:-20]:
        train = train.append(pd.read_csv(path + file))

    # Append last 20 files to test frame
    for file in csv_files[-20:]:
        test = test.append(pd.read_csv(path + file))

    # Output files to csv directory in order to do feature engineering
    train.to_csv('csv/' + directory + '_train.csv', index=False)
    test.to_csv('csv/' + directory + '_test.csv', index=False)

# Merge benchmarks
merge(directory='A1Benchmark')
merge(directory='A2Benchmark')
merge(directory='A3Benchmark')
merge(directory='A4Benchmark')