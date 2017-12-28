from keras.models import Sequential
from keras import layers
import numpy as np

from .dataset import CardImgGenerator


class CardFinder(object):
    def __init__(self):
        self.model = None
        self.dataset_provider = CardImgGenerator()
        self.img_size = (600, 450)

        self.input_dim = self.img_size[0] * self.img_size[1] * 3

    def init_model(self):
        self.model = Sequential([
            layers.Dense(200, input_dim=self.input_dim),
            layers.Activation('relu'),
            layers.Dropout(0.2),
            layers.Dense(4)
        ])
        self.model.compile('adadelta', 'mse')

        self.load_weights()

    def save_weights(self):
        self.model.save_weights(self.get_weights_path())

    def load_weights(self):
        """Loads model weights if file exists"""
        try:
            self.model.load_weights(self.get_weights_path())
            print('Loaded weights')
            return True
        except OSError:
            return False

    def get_weights_path(self):
        return f'saved/{self.__class__.__name__}.h5'

    def train(self, data_amount, epochs):
        batch_x, batch_y = self.dataset_provider.get_batch(data_amount)
        batch_x = batch_x.reshape(-1, self.input_dim)
        batch_y = batch_y / (*self.img_size, *self.img_size)

        divider = int(0.8 * data_amount)
        train_x = batch_x[:divider]
        test_x = batch_x[divider:]
        train_y = batch_y[:divider]
        test_y = batch_y[divider:]

        result = self.model.fit(train_x, train_y, epochs=epochs, validation_data=(test_x, test_y), verbose=2)

        return result
