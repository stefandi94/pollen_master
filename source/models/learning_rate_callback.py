from keras.callbacks import Callback
import keras.backend as K


class LearningRateCallback(Callback):

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)

    # def on_epoch_end(self, epoch, logs=None):
    #     print(K.eval(self.model.optimizer.lr))
    # def on_epoch_end(self, epoch, logs=None):
    #     lr = self.model.optimizer.lr
    #     decay = self.model.optimizer.decay
    #     iterations = self.model.optimizer.iterations
    #     lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
    #     print(K.eval(lr_with_decay))