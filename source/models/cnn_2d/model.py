from typing import Any

from keras.layers import concatenate, Dense, Input, LeakyReLU, Dropout
from keras.models import Model

from source.base_dl_model import BaseDLModel
from source.models.convolutional_layers.cnn_layers import create_cnn_network
from source.models.dense_layers import create_dense_network


class CNNModel(BaseDLModel):
    convolution_filters = [64, 64, 128, 128, 256, 512]

    def __init__(self,
                 **parameters: Any) -> None:
        super().__init__(**parameters)

    def build_model(self) -> None:
        inputs = [Input(input_shape) for input_shape in self.input_shape]

        layers = [create_cnn_network(layer, self.convolution_filters, kernels=(2, 2),
                                     dropout=0.2, batch_normalization=True, flatting=False) for layer in inputs]
        layers = [LeakyReLU()(layer) for layer in layers]
        layers = [create_dense_network(layer, num_of_neurons=[200]) for layer in layers]
        layer = concatenate([layer for layer in layers])

        layer = Dropout(0.3)(layer)
        output = Dense(self.num_classes, activation='softmax')(layer)

        model = Model(inputs, output)
        self.model = model

    def __str__(self):
        return 'CNN'
