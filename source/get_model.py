from source.models import LSTMModel, CNNModel


def get_model(model_name):
    model_name = model_name.lower()
    if model_name == 'lstm':
        model = LSTMModel
    elif model_name == 'cnn':
        model = CNNModel
    else:
        raise Exception(f'Model name {model_name} does not exist! Please choose from CNN_1D or CNN_2D')

    return model
