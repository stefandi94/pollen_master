from typing import List, Tuple

import keras
from keras.engine import Layer
from keras.initializers import Initializer
from keras.layers import Conv2D, BatchNormalization, GlobalAvgPool2D, Conv1D, Activation, Dropout, LeakyReLU, Flatten
from keras.regularizers import Regularizer

kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)


def create_cnn_layer(input_layer: "Layer",
                     num_filter: int,
                     kernel_size: int or Tuple[int, int] = (3, 3),
                     dropout: float = 0.0,
                     batch_normalization: bool = False,
                     kernel_init: str or "Initializer" = kernel_init,
                     bias_init: str or "Initializer" = bias_init,
                     strides: int or Tuple[int, int] = (1, 1),
                     kernel_regularizer: Regularizer = None,
                     activation: bool = True) -> "Layer":
    """
    Given input layer and number of filters, do 2D convolution
    :param input_layer: Input layer
    :param num_filter: Number of feature maps
    :param batch_normalization
    :param dropout:
    :param kernel_init:
    :param bias_init:
    :param kernel_size:
    :param strides
    :param kernel_regularizer
    :param activation
    :return: Layer
    """
    conv = Conv2D
    layer = conv(num_filter,
                 strides=strides,
                 kernel_size=kernel_size,
                 padding='same',
                 kernel_initializer=kernel_init,
                 bias_initializer=bias_init,
                 kernel_regularizer=kernel_regularizer)(input_layer)

    if batch_normalization:
        layer = BatchNormalization()(layer)
    if activation:
        layer = LeakyReLU()(layer)

    layer = Dropout(dropout)(layer)

    return layer


def create_cnn_network(input_layer: "Layer",
                       num_of_filters: List[int],
                       dropout: float = 0,
                       batch_normalization: bool = False,
                       kernels: List[int] or List[Tuple] = None,
                       flatting: bool = False) -> "Layer":
    """
    Given input layer and number of filters, creates network
    :param input_layer:
    :param num_of_filters:
    :param kernels:
    :param dropout
    :param batch_normalization
    :param flatting: Flattening or GlobalAvgPooling
    :return:
    """
    if kernels is None:
        kernels = (2, 2)

    layer = input_layer
    for index, filter in enumerate(num_of_filters):
        layer = create_cnn_layer(input_layer=layer,
                                 num_filter=filter,
                                 kernel_size=kernels,
                                 dropout=dropout,
                                 batch_normalization=batch_normalization)
    if flatting:
        layer = Flatten()(layer)
    else:
        layer = GlobalAvgPool2D()(layer)
    return layer
