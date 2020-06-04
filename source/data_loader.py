import os
from typing import List

from settings import NS_STANDARDIZED_TRAIN_DIR, NS_NORMALIZED_VALID_DIR, NS_NORMALIZED_TEST_DIR, \
    NS_NORMALIZED_TRAIN_DIR, NS_STANDARDIZED_TEST_DIR, NS_STANDARDIZED_VALID_DIR, NS_DATA_DIR
from source.data_reader import load_all_data, create_3d_array, create_4d_array
from utils.split_data import save_data, load_data
from utils.utilites import calculate_weights, get_features


def data(standardized: bool,
         train: bool,
         save_dir: str,
         create_4d_arr: bool = False,
         features: List[int] = [0, 1, 2]):
    """
    Load data for training or testing model and convert it to right format
    :param standardized: Whether to load standardized data if true, otherwise load normalized data
    :param train: Should load data for training or predicting
    :param save_dir: If model is not training, path to weights of classes and mapping of classes of already trained model
    :param create_4d_arr: Whether to convert data to 4d format (images)
    :param features
    :return:
    """
    if standardized:
        TRAIN_DIR = NS_NORMALIZED_TRAIN_DIR
        VALID_DIR = NS_NORMALIZED_VALID_DIR
        TEST_DIR = NS_NORMALIZED_TEST_DIR
    else:
        TRAIN_DIR = NS_STANDARDIZED_TRAIN_DIR
        VALID_DIR = NS_STANDARDIZED_VALID_DIR
        TEST_DIR = NS_STANDARDIZED_TEST_DIR

    if train:
        X_train, y_train = load_all_data(TRAIN_DIR)
        X_valid, y_valid = load_all_data(VALID_DIR)

        X_train.pop(1)  # remove 1x1 feature
        X_valid.pop(1)
        X_train = X_train[:3]
        X_valid = X_valid[:3]

        X_train = get_features(X_train, features=features)
        X_valid = get_features(X_valid, features=features)

        for index in range(len(X_train)):
            if len(X_train[index]) < 3:
                X_train[index] = create_3d_array(X_train[index])

            if len(X_valid[index]) < 3:
                X_valid[index] = create_3d_array(X_valid[index])

        if create_4d_arr:
            for index in range(len(X_train)):
                X_train[index] = create_4d_array(X_train[index])
                X_valid[index] = create_4d_array(X_valid[index])

        weight_class = calculate_weights(y_train)
        if not os.path.exists(os.path.join(NS_DATA_DIR, 'weight_class')):
            save_data(weight_class, NS_DATA_DIR, 'weight_class')


        return X_train, y_train, X_valid, y_valid, weight_class

    else:
        X_test, y_test = load_all_data(TEST_DIR)
        X_test.pop(1)
        X_test = X_test[:3]

        for index in range(len(X_test)):
            if len(X_test[index]) < 3:
                X_test[index] = create_3d_array(X_test[index])

        if create_4d_arr:
            for index in range(len(X_test)):
                X_test[index] = create_4d_array(X_test[index])

        label_to_index = load_data(NS_DATA_DIR, 'label_to_index')
        weight_class = load_data(NS_DATA_DIR, 'weight_class')
        X_test = get_features(X_test, features=features)

        return X_test, y_test, weight_class, label_to_index
