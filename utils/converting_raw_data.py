#!/usr/bin/env python
# coding: utf-8
import json
import os
import os.path as osp
import pickle
from collections import Counter

import numpy as np
import pandas as pd

from settings import NS_RAW_DATA_DIR, NS_DATA_DIR, OS_DATA_DIR, OS_RAW_DATA_DIR
from utils.preprocessing import label_to_index, calculate_and_check_shapes
from utils.utilites import count_values


def transform_raw_data(raw_data_path, classes_to_take=None):
    files = sorted(os.listdir(raw_data_path))
    data = {"scatter": [], "size": [], "life_1": [], "spectrum": [], "life_2": []}
    enocoded_labels = []

    class_to_num, string_labels = label_to_index(files)

    for file_name in files:
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
    with open('label_dict_ns.json', 'r') as fp:
        label_dict_ns = json.load(fp)

    label_dict_ns = dict((key, int(value)) for key, value in label_dict_ns.items())

    ns_most_common = find_most_common(label_dict_ns)
    most_common_labels = [label[0] for label in ns_most_common]
    data, enocoded_labels, string_labels, class_to_num, feature_names = transform_raw_data(NS_RAW_DATA_DIR, most_common_labels)
    print()
    # num_to_class = dict((value, key) for key, value in class_to_num.items())
    # encoded_label_dict = count_values(enocoded_labels)
    # string_label_dict = dict()
    #
    # for key, value in encoded_label_dict.items():
    #     string_label_dict[num_to_class[key]] = str(value)
    #
    # with open('label_dict_ns.json', 'w') as fp:
    #     json.dump(string_label_dict, fp)
    #
    # class_to_num = dict_int_to_string(class_to_num)
    #
    # with open('class_to_num_ns.json', 'w') as fp:
    #     json.dump(class_to_num, fp)
    # print()
    # with open('label_dict_ns.json', 'r') as fp:
    #     label_dict_ns = json.load(fp)
    #
    # with open('label_dict_os.json', 'r') as fp:
    #     label_dict_os = json.load(fp)
    #
    # ns_most_common = find_most_common(label_dict_ns)
    # os_most_common = find_most_common(label_dict_os)
    #
    with open('ns_most_common.json', 'w') as fp:
        json.dump(ns_most_common, fp)
    # with open('os_most_common.json', 'w') as fp:
    #     json.dump(os_most_common, fp)

    # with open(osp.join(NS_DATA_DIR, 'label_to_index_os.pckl'), 'wb') as handle:
    #     pickle.dump(class_to_num, handle, protocol=pickle.HIGHEST_PROTOCOL)
