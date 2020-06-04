from keras.utils import to_categorical

from settings import NUM_OF_CLASSES
from source.data_loader import data
from source.get_model import get_model
from source.plotting_predictions import plot_confusion_matrix, create_dict_conf, plot_confidence, plot_classes, \
    plot_confidence_per_class, plot_history
from utils.utilites import smooth_labels, calculate_input_shapes

parameters = {
    'model': 'CNN',
    'epochs': 3000,
    'batch_size': 128,
    'learning_rate': 0.005,
    'smooth_factor': 0.08,
    'train_model': True,
    'save_dir': './model_weights/real_normalized/CNN/2_3_features/64_64_128_128_256_512_flatten',
    'load_dir': None,
    # 'load_dir': './model_weights/real_normalized/CNN/2_3_features/512_flatten/795-1.122-0.779-0.927-0.711.hdf5',
    'standardized': True,
    'show_plots': False,
    '4d_array': True,
    'features': [1, 2],
}


def train(parameters):
    if parameters['train_model']:
        X_train, y_train, X_valid, y_valid, weight_class = data(standardized=parameters['standardized'],
                                                                train=parameters['train_model'],
                                                                save_dir=parameters['save_dir'],
                                                                create_4d_arr=parameters['4d_array'],
                                                                features=parameters['features'])

        input_shape = calculate_input_shapes(X_train, input_4d=parameters['4d_array'])
        parameters['input_shape'] = input_shape
        y_train_cate = to_categorical(y_train, NUM_OF_CLASSES)
        y_valid_cate = to_categorical(y_valid, NUM_OF_CLASSES)

        smooth_labels(y_train_cate, parameters['smooth_factor'])
        dl_model = get_model(parameters['model'])(**parameters)
        dl_model.train(X_train,
                       y_train_cate,
                       X_valid,
                       y_valid_cate,
                       weight_class=weight_class,
                       lr_type='cosine')

    else:
        X_test, y_test, weight_class, dict_mapping = data(standardized=parameters['standardized'],
                                                          train=parameters['train_model'],
                                                          save_dir=parameters['save_dir'],
                                                          create_4d_arr=parameters['4d_array'],
                                                          features=parameters['features'])
        y_test_cate = to_categorical(y_test, NUM_OF_CLASSES)

        dl_model = get_model(parameters['model'])(**parameters)
        dl_model.load_model(path=parameters['load_dir'])

        y_pred = dl_model.predict(X_test)
        evaluation = dl_model.model.evaluate(X_test, y_test_cate, batch_size=parameters['batch_size'])
        print(f'Accuracy is {evaluation[1]}')

        inverse_dict_mapping = {v: k for k, v in dict_mapping.items()}

        y_pred_real = [inverse_dict_mapping[y[0]] for y in y_pred]
        y_test_real = [inverse_dict_mapping[y] for y in y_test]

        plot_confusion_matrix(y_test_real, y_pred_real, classes=list(inverse_dict_mapping.values()),
                              path=parameters['save_dir'], show_plot=parameters['show_plots'])

        true_conf, true_dicti, false_conf, false_dicti = create_dict_conf(y_test, y_pred, NUM_OF_CLASSES)
        plot_confidence(true_conf, false_conf, parameters['save_dir'], show_plot=parameters['show_plots'])

        plot_classes(y_test, y_pred, parameters['save_dir'], classes=list(inverse_dict_mapping.values()),
                     show_plot=parameters['show_plots'])

        plot_confidence_per_class(true_dicti, false_dicti, list(inverse_dict_mapping.values()),
                                  parameters['save_dir'], show_plot=parameters['show_plots'])

        plot_history(log_path=parameters['save_dir'], show_plot=parameters['show_plots'])
        print()


if __name__ == '__main__':
    train(parameters=parameters)
