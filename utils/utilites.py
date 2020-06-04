from typing import List

import numpy as np
from sklearn.utils import class_weight


def smooth_labels(y: np.ndarray,
                  smooth_factor: float) -> np.ndarray:
    """Convert a matrix of one-hot row-vector labels into smoothed versions.
    # Arguments
        y: matrix of one-hot row-vector labels to be smoothed
        smooth_factor: label smoothing factor (between 0 and 1)
    # Returns
        A matrix of smoothed labels.
    """

    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        raise Exception(
            'Invalid label smoothing factor: ' + str(smooth_factor))
    return y


def calculate_weights(y_train: np.ndarray or List[int]) -> np.ndarray:
    """
    Given class labels of training sample, calculates frequency of each class and return inverted proportion, to
    balance dataset.
    :param y_train: training labels
    :return: weight class for each label
    """

    weight_class = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    return weight_class


def count_values(array):
    """
    Given array return dictionary with class numbers and number of instances of that class
    :param array:
    :return:
    """
    unique, counts = np.unique(array, return_counts=True)
    return dict(zip(unique, counts))


def get_features(X, features: List[int]):
    new_X = []
    for index, x in enumerate(X):
        if index in features:
            new_X.append(x)
    return new_X


def calculate_input_shapes(X, input_4d=True):
    """
    :param X:
    :param input_4d: Create input for CNN
    :return:
    """
    input_shape = []
    for x in X:
        if input_4d:
            input_shape.append((x.shape[1], x.shape[2], 1))
        else:
            input_shape.append((x.shape[1], x.shape[2]))
    return input_shape
