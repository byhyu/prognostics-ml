from os import listdir

from os.path import isfile, join

'''
ConvertToCSV
------------

Convert all .txt files which contains data and
split into train and test files.

Input directory  : dataset
Output directory : csv
'''
input_directory = 'dataset/'
output_directory = 'csv/'

txt_files = [f for f in listdir(input_directory) if
             isfile(join(input_directory, f)) and f.endswith('.txt') and f <> 'readme.txt'][::-1]

for in_file in txt_files:
    out_file = in_file[:-3]
    out_file = out_file + 'csv'
    out_file = open(join(output_directory, out_file), 'a')

    # Read lines from input file
    lines = open(join(input_directory, in_file), 'rb').readlines()

    # Create header for the dataset
    header = ""
    if 'RUL' in in_file:
        header = 'RUL\n'
    elif 'test' or 'train' in in_file:
        header = 'UnitNumber,Time,Setting1,Setting2,Setting3,Sensor1,Sensor2,Sensor3,Sensor4,Sensor5,Sensor6,Sensor7,Sensor8,Sensor9,Sensor10,Sensor11,Sensor12,Sensor13,Sensor14,Sensor15,Sensor16,Sensor17,Sensor18,Sensor19,Sensor20,Sensor21\n'
    out_file.write(header)

    # Convert data in to csv format
    for line in lines:
        line = list(line.strip("\n ").split(" "))
        line = ','.join(line)
        line += '\n'
        out_file.write(line)

    out_file.close()

print "Conversion complete"
