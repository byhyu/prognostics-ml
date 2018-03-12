import xml.etree.ElementTree as ET
from featureeng.Container import Frame
import pandas as pd

# features
MOVING_AVERAGE = 'moving_average'
MOVING_THRESHOLD_AVERAGE = 'moving_threshold_average'
MOVING_MEDIAN_CENTERED_AVERAGE = 'moving_median_centered_average'
MOVING_K_CLOSEST_AVERAGE = 'moving_k_closest_average'
MOVING_MEDIAN = 'moving_median'
MOVING_STANDARD_DEVIATION = 'moving_standard_deviation'
MOVING_ENTROPY = 'moving_entropy'
MOVING_PROBABILITY = 'moving_probability'
MOVING_VARIANCE = 'moving_variance'

# attributes
WINDOW = 'window'
THRESHOLD = 'threshold'
BOUNDRY = 'boundry'
K_CLOSEST = 'k_closest'
NO_OF_BINS = 'no_of_bins'


def apply_feature_eng(frame, xml_file):
    if isinstance(frame, pd.core.frame.DataFrame):
        frame = Frame(frame)
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for feature in root:
        if feature.tag == MOVING_AVERAGE:
            window = int(feature.get(WINDOW))
            for column in feature:
                column = str(column.text)
                frame.apply_moving_average(input_column=column, window=window)

        elif feature.tag == MOVING_THRESHOLD_AVERAGE:
            window = int(feature.get(WINDOW))
            threshold = float(feature.get(THRESHOLD))
            for column in feature:
                column = str(column.text)
                frame.apply_moving_threshold_average(input_column=column, window=window, threshold=threshold)

        elif feature.tag == MOVING_MEDIAN_CENTERED_AVERAGE:
            window = int(feature.get(WINDOW))
            boundry = int(feature.get(BOUNDRY))
            for column in feature:
                column = str(column.text)
                frame.apply_moving_median_centered_average(input_column=column, window=window, boundary=boundry)

        elif feature.tag == MOVING_K_CLOSEST_AVERAGE:
            window = int(feature.get(WINDOW))
            kclosest = int(feature.get(K_CLOSEST))
            for column in feature:
                column = str(column.text)
                frame.apply_moving_k_closest_average(input_column=column, window=window, kclosest=kclosest)

        elif feature.tag == MOVING_MEDIAN:
            window = int(feature.get(WINDOW))
            for column in feature:
                column = str(column.text)
                frame.apply_moving_median(input_column=column, window=window)

        elif feature.tag == MOVING_STANDARD_DEVIATION:
            window = int(feature.get(WINDOW))
            for column in feature:
                column = str(column.text)
                frame.apply_moving_std(input_column=column, window=window)

        elif feature.tag == MOVING_ENTROPY:
            window = int(feature.get(WINDOW))
            no_of_bins = int(feature.get(NO_OF_BINS))
            for column in feature:
                column = str(column.text)
                frame.apply_moving_entropy(input_column=column, window=window, no_of_bins=no_of_bins)

        elif feature.tag == MOVING_PROBABILITY:
            window = int(feature.get(WINDOW))
            no_of_bins = int(feature.get(NO_OF_BINS))
            for column in feature:
                column = str(column.text)
                frame.apply_moving_entropy(input_column=column, window=window, no_of_bins=no_of_bins)

        elif feature.tag == MOVING_VARIANCE:
            window = int(feature.get(WINDOW))
            for column in feature:
                column = str(column.text)
                frame.apply_moving_variance(input_column=column, window=window)

    return frame.get_panda_frame()
