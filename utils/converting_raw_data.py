#!/usr/bin/env python
# coding: utf-8
import json
import os
import os.path as osp
import pickle

import numpy as np
import pandas as pd

from settings import NS_RAW_DATA_DIR, NS_DATA_DIR, OS_DATA_DIR, OS_RAW_DATA_DIR
from utils.preprocessing import label_to_index, calculate_and_check_shapes


def transform_raw_data(raw_data_path):
    files = sorted(os.listdir(raw_data_path))
    data = {"scatter": [], "size": [], "life_1": [], "spectrum": [], "life_2": []}
    labels = []

    class_to_num = label_to_index(files)

    for file_name in files:
        if file_name.split(".")[-1] != "json":
            continue

        raw_data = json.loads(open(osp.join(raw_data_path, file_name)).read())

        for i in range(len(raw_data["Data"])):
            specmax = np.max(raw_data["Data"][i]["Spectrometer"])
            file_data = raw_data["Data"][i]
            calculate_and_check_shapes(file_data, file_name, specmax, data, labels, class_to_num)

    feature_names = ["scatter", "size", "life_1", "spectrum", "life_2"]
    return data, labels, class_to_num, feature_names


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


if __name__ == '__main__':
    data, labels, label_to_index, feature_names = transform_raw_data(OS_RAW_DATA_DIR)
    print()
    # with open(osp.join(NS_DATA_DIR, 'label_to_index.pckl'), 'wb') as handle:
    #     pickle.dump(label_to_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
