import argparse

from card_finder.model import CardFinder

EPOCHS = 10000
STEP = 1000
DATA_AMOUNT = 100

parser = argparse.ArgumentParser()

parser.add_argument('epochs', default=EPOCHS, nargs='?',
                    help=f'Amount of epochs to train. Default: {EPOCHS}')
parser.add_argument('step', default=STEP, nargs='?',
                    help=f'Amount of train epochs before saving weights. Default: {STEP}')
parser.add_argument('data_amount', default=DATA_AMOUNT, nargs='?',
                    help=f'Amount of data to train with each epoch. Default: {DATA_AMOUNT}')

if __name__ == '__main__':
    args = parser.parse_args()

    sf = CardFinder()
    sf.init_model()
    for i in range(args.epochs//args.step):
        sf.train(args.data_amount, args.step)
        sf.save_weights()
