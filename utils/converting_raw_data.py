#!/usr/bin/env python
# coding: utf-8
import json
import os
import os.path as osp
from collections import Counter

import numpy as np
import pandas as pd

from settings import NS_RAW_DATA_DIR, POLLEN_TYPES, NEW_RAW_DATA, NEW_DATA
from utils.preprocessing import label_to_index, calculate_and_check_shapes


def transform_raw_data(raw_data_path, classes_to_take=None):
    files = sorted(os.listdir(raw_data_path))
    data = {"scatter": [], "size": [], "life_1": [], "spectrum": [], "life_2": []}
    enocoded_labels = []

    class_to_num, string_labels = label_to_index(files)

    for index, file_name in enumerate(files):
        if index % 2 == 0:
            print(f'Current file is {file_name}')

        if file_name.split(".")[-1] != "json":
            continue

        if classes_to_take is not None:
            if not file_name.split(".")[0] in classes_to_take:
                continue

        raw_data = json.loads(open(osp.join(raw_data_path, file_name)).read())

        for i in range(len(raw_data["Data"])):
            specmax = np.max(raw_data["Data"][i]["Spectrometer"])
            file_data = raw_data["Data"][i]
            calculate_and_check_shapes(file_data, file_name, specmax, data, enocoded_labels, class_to_num)

    feature_names = ["scatter", "size", "life_1", "spectrum", "life_2"]
    return data, enocoded_labels, string_labels, class_to_num, feature_names


def create_lifetime(data, path_to_save):
    lista = []
    for i in range(len(data[2])):

        if i == 0:
            l = ["Cupressus"]
        elif i == 1:
            l = ["Fraxinus excelsior"]
        else:
            l = ["Ulmus"]

        if np.where(data[2][i][0, :] == np.max(data[2][i][0, :]))[0].shape[0] > 1:
            l.append("Yes")
        else:
            l.append("No")

        for k in range(4):
            if k != 2:
                l.append(np.max(data[2][i][k, :]) / np.e)
        lista.append(l)

    features = ["Pollen type", "Saturated", "Time of band 1", "Time of band 2", "Time of band 3"]
    amb = pd.DataFrame(columns=features)

    for i in range(len(lista)):
        amb.loc[len(amb)] = lista[i]
    amb.to_csv(osp.join(path_to_save, "Time of lifetime.csv"), index=False)


def dict_int_to_string(my_dict):
    return dict((str(k), str(v)) for k, v in my_dict.items())


def find_most_common(label_dict, n=10):
    label_counter = Counter(label_dict)
    most_common = label_counter.most_common(n)
    return [label for label in most_common]


if __name__ == '__main__':
    class_to_num, string_labels = transform_raw_data(NEW_RAW_DATA)
    data, enocoded_labels, string_labels, class_to_num, feature_names = transform_raw_data(NEW_RAW_DATA)
    print()
