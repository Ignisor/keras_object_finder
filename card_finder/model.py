from keras.models import Sequential
from keras import layers, optimizers

from .dataset import CardImgGenerator


class CardFinder(object):
    def __init__(self):
        self.model = None
        self.dataset_provider = CardImgGenerator()
        self.img_size = (600, 450)

        self.input_shape = (self.img_size[0], self.img_size[1], 3)

    def init_model(self):
        self.model = Sequential([
            layers.Conv2D(32, (3, 3), padding='same', input_shape=self.input_shape),
            layers.Activation('relu'),
            layers.Conv2D(32, (3, 3)),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(32, (3, 3), padding='same'),
            layers.Activation('relu'),
            layers.Conv2D(32, (3, 3)),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(512),
            layers.Activation('relu'),
            layers.Dropout(0.5),

            layers.Dense(4),
            layers.Activation('softmax')
        ])

        opt = optimizers.rmsprop(lr=1e-4, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

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
        train_amount = int(data_amount*0.8)
        test_amount = int(data_amount*0.2)

        train_batch = self.dataset_provider.get_batch()
        test_batch = self.dataset_provider.get_batch()

        result = self.model.fit_generator(
            train_batch,
            steps_per_epoch=train_amount,
            epochs=epochs,
            validation_data=test_batch,
            validation_steps=test_amount,
            verbose=2
        )

        return result
