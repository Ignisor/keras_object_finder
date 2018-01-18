from keras.models import Sequential
from keras import layers, constraints

from .dataset import CardImgGenerator


class CardFinder(object):
    def __init__(self):
        self.model = None
        self.dataset_provider = CardImgGenerator()
        self.img_size = (128, 128)

        self.input_shape = (self.img_size[0], self.img_size[1], 3)

        self.output_shape = (10, 10)

    def init_model(self):
        self.model = Sequential([
            layers.Conv2D(32, (5, 5), padding='same', input_shape=self.input_shape, activation='relu',
                          data_format="channels_last"),
            layers.Conv2D(32, (5, 5), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
            layers.Conv2D(32, (5, 5), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),

            layers.Dense(self.output_shape[0] * self.output_shape[1], activation='softmax'),

            # layers.Reshape(self.output_shape)
        ])

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

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

        train_batch = self.dataset_provider.get_generator()
        test_batch = self.dataset_provider.get_generator()

        result = self.model.fit_generator(
            train_batch,
            steps_per_epoch=train_amount,
            epochs=epochs,
            validation_data=test_batch,
            validation_steps=test_amount,
            verbose=2
        )

        return result
