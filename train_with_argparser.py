from settings import NUM_OF_CLASSES
from train import train

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='Path to input data')
    parser.add_argument('-tm', '--train_model', default=False, help='Path to input data')
    parser.add_argument('-e', '--epochs', default=1000, help='Number of epochs to train')
    parser.add_argument('-bs', '--batch_size', default=64, help='Batch size number')
    parser.add_argument('-lr', '--learning_rate', default=0.005, help='Learning rate of model')
    parser.add_argument('-sf', '--smooth_factor', default=0.1, help="Smooth factor for training labels")
    parser.add_argument('-sd', '--save_dir', help='Number of classes',
                        default='./model_weights/LSTM/3_features')
    parser.add_argument('-ld', '--load_dir', required=False, help="Path to load dir")
                        # default='./model_weights/CNN_2D/leaky_relu/985-1.855-0.819-1.120-0.764.hdf5')
    parser.add_argument('-st', '--standardized', default=True, help="Boolean if to use standardized data")
    parser.add_argument('-sp', '--show_plots', default=True, help="Whether to print plots")

    args = parser.parse_args()
    model = args.model
    train_model = args.train_model
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    smooth_factor = args.smooth_factor
    save_dir = args.save_dir
    load_dir = args.load_dir
    standardized = args.standardized
    show_plots = args.show_plots

    parameters = {
        'model': model,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'smooth_factor': smooth_factor,
        'train_model': train_model,
        'save_dir': save_dir,
        'load_dir': load_dir,
        'standardized': standardized,
        'show_plots': show_plots,
    }

    train(parameters)
