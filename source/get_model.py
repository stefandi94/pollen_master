from source.models import CNN_1D, CNN_2D


def get_model(model_name):
    if model_name == 'CNN_1D':
        model = CNN_1D
    elif model_name == 'CNN_2D':
        model = CNN_2D
    else:
        raise Exception(f'Model name {model_name} does not exist! Please choose from CNN_1D or CNN_2D')

    return model
