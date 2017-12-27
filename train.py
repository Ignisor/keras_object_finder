from card_finder.model import CardFinder


if __name__ == '__main__':
    sf = CardFinder()
    sf.init_model()
    sf.train(100, 10000)
