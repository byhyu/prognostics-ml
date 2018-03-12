import pandas as pd
import h2o
import numpy as np
from sklearn_pandas import DataFrameMapper


def pandasToH2O(panda_frame):
    h2o.init()
    parsed_frame = h2o.H2OFrame(panda_frame)
    parsed_frame.set_names(list(panda_frame.columns))
    return parsed_frame


def pandasToSkLearn(panda_frame, response_column, seperate_target=False):
    input_columns = list(panda_frame.columns)
    input_columns.remove(response_column)
    df_mapper = DataFrameMapper([(input_columns, None), (response_column, None)])
    parsed_frame = df_mapper.fit_transform(panda_frame)

    if seperate_target:
        column_count = len(list(panda_frame.columns))
        yield parsed_frame[:, 0:column_count-1]
        yield parsed_frame[:, column_count-1]
    else:
        yield parsed_frame


def h2oToNumpyArray(h2o_frame):
    h2o_frame = h2o_frame.get_frame_data()
    return np.array(map(float, h2o_frame.split("\n")[1:-1]))


def h2oToList(h2o_frame):
    h2o_frame = h2o_frame.get_frame_data()
    return h2o_frame.split("\n")[1:-1]
