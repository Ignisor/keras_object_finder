import argparse

from card_finder.model import CardFinder

EPOCHS = 10000
STEP = 1000
DATA_AMOUNT = 100

parser = argparse.ArgumentParser()

parser.add_argument('epochs', default=EPOCHS, nargs='?', type=int,
                    help=f'Amount of epochs to train. Default: {EPOCHS}')
parser.add_argument('step', default=STEP, nargs='?', type=int,
                    help=f'Amount of train epochs before saving weights. Default: {STEP}')
parser.add_argument('data_amount', default=DATA_AMOUNT, nargs='?', type=int,
                    help=f'Amount of data to train with each epoch. Default: {DATA_AMOUNT}')


def train(model, epochs, step, data_amount):
    for _ in range(epochs // step):
        model.train(data_amount, step)
        model.save_weights()


if __name__ == '__main__':
    args = parser.parse_args()

    m = CardFinder()
    m.init_model()

    try:
        train(m, args.epochs, args.step, args.data_amount)
    except KeyboardInterrupt as e:
        m.save_weights()
        raise e


