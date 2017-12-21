from keras.models import Sequential
from keras import layers

from .dataset import SquareGenerator


class SquareFinder(object):
    def __init__(self):
        self.model = None
        self.dataset_provider = SquareGenerator()
        self.img_size = 8

        self.input_dim = self.img_size ** 2

    def init_model(self):
        self.model = Sequential([
            layers.Dense(200, input_dim=self.input_dim),
            layers.Activation('relu'),
            layers.Dropout(0.2),
            layers.Dense(4)
        ])
        self.model.compile('adadelta', 'mse')

    def train(self, data_amount, epochs):
        batch_x, batch_y = self.dataset_provider.get_batch(data_amount)
        batch_x = batch_x.reshape(-1, self.input_dim)
        batch_y = batch_y / self.img_size

        divider = int(0.8 * data_amount)
        train_x = batch_x[:divider]
        test_x = batch_x[divider:]
        train_y = batch_y[:divider]
        test_y = batch_y[divider:]

        return self.model.fit(train_x, train_y, epochs=epochs, validation_data=(test_x, test_y), verbose=2)
