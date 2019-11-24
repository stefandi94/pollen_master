import json

from keras.utils import to_categorical

from source.data_loader import data
from source.models import CNN_2D
from utils.utilites import smooth_labels

smooth_factor = 0.5
shapes = dict(input_shape_1=(20, 120),
              input_shape_2=(4, 24),
              input_shape_3=(4, 32))

standardized = True
normalized = False
NUM_OF_CLASSES = 10

# load_dir = '/mnt/hdd/PycharmProjects/pollen_classification/new_weights/ns/normalized/smooth_factor_0.1/optimizer_adam' \
#            '/learning_rate_type_cosine/model_name_CNNRNN/ '
# load_dir = '/mnt/hdd/PycharmProjects/pollen_classification/new_weights/ns/standard_normal/smooth_factor_0.0/' \
#            'optimizer_adam/learning_rate_type_cosine/model_name_ANN/'
save_dir = './model_weights/CNN_2D'

parameters = {'epochs': 30,
              'batch_size': 64,
              'optimizer': 'adam',
              'num_classes': NUM_OF_CLASSES,
              'save_dir': save_dir}
# 'load_dir': f'{os.path.join(load_dir, "8/25-0.872-0.713-0.949-0.694.hdf5")}'}

if __name__ == '__main__':
    with open('./utils/ns_most_common.json', 'r') as fp:
        classes = json.load(fp)

    X_train, y_train, X_valid, y_valid, X_test, y_test, weight_class, dict_mapping = data(standardized=True,
                                                                                          ns=True,
                                                                                          num_of_classes=NUM_OF_CLASSES,
                                                                                          create_4d_arr=True)
    y_train_cate = to_categorical(y_train, NUM_OF_CLASSES)
    y_valid_cate = to_categorical(y_valid, NUM_OF_CLASSES)
    y_test_cate = to_categorical(y_test, NUM_OF_CLASSES)

    smooth_labels(y_train_cate, smooth_factor)
    #
    dnn = CNN_2D(**parameters)
    # dnn.load_model(parameters["load_dir"])
    dnn.train(X_train,
              y_train_cate,
              X_valid,
              y_valid_cate,
              weight_class=weight_class,
              lr_type='cyclic')
    print()
    # import os.path as osp
    # os.makedirs('./test', exist_ok=True)
    # # with open(osp.join('./test', 'mapping.pckl'), 'wb') as handle:
    # #     pickle.dump(dict_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    y_pred = dnn.predict(X_test)
    eval_1 = dnn.model.evaluate(X_train, y_train_cate, batch_size=64)
    eval_2 = dnn.model.evaluate(X_valid, y_valid_cate, batch_size=64)
    eval_3 = dnn.model.evaluate(X_test, y_test_cate, batch_size=64)
    print()
    # # real_y = [dict_mapping[y] for y in y_test]
    # # pred_y = [(dict_mapping[y[0]], y[1]) for y in y_pred]
    #
    # print(f'Accuracy is {eval[1]}')
    # y_class_pred = [int(pred[0]) for pred in y_pred]
    # conf_matrix = confusion_matrix(y_test, y_class_pred)
    # true_conf, true_dicti, false_conf, false_dicti = create_dict_conf(y_test, y_pred, NUM_OF_CLASSES)
    #
    # # plot_confusion_matrix(y_test, y_class_pred, list(dict_mapping.keys()), './test')
    # # plot_confusion_matrix(y_test, y_class_pred, list(dict_mapping.keys()), parameters['save_dir'], normalize=True)
    # plot_confidence(true_conf, false_conf, parameters['save_dir'], show_plot=False)
    # plot_classes(y_test, y_pred, parameters['save_dir'], NUM_OF_CLASSES, show_plot=False)
    # plot_confidence_per_class(true_dicti, false_dicti, NUM_OF_CLASSES, parameters['save_dir'], show_plot=False)
    # plot_history(log_path=parameters['save_dir'])
