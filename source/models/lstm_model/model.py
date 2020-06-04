from typing import Any

from keras import Input, Model
from keras.layers import concatenate, Dense, LSTM, Dropout

from source.base_dl_model import BaseDLModel


class LSTMModel(BaseDLModel):
    rnn_filters = [512]

    def __init__(self, **parameters: Any) -> None:
        super().__init__(**parameters)

    def build_model(self) -> None:
        inputs = [Input(input_shape) for input_shape in self.input_shape]

        layers = [LSTM(512, recurrent_dropout=0.2, dropout=0.2)(layer) for layer in inputs]
        layers = [Dense(128)(layer) for layer in layers]
        layer = concatenate([layer for layer in layers])
        layer = Dropout(0.2)(layer)
        output = Dense(self.num_classes, activation='softmax')(layer)

        model = Model(inputs, output)
        self.model = model

    def __str__(self):
        return 'LSTM'
