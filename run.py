import numpy as np

from card_finder.model import CardFinder


if __name__ == '__main__':
    sf = CardFinder()
    sf.init_model()
    sf.train(100, 10000)

    tx, ty = sf.dataset_provider.get_batch(10)
    py = sf.model.predict(tx.reshape(-1, sf.input_dim))
    pb = py * (*sf.img_size, *sf.img_size)
