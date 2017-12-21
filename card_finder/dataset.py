import os
import random

import numpy as np
from PIL import Image

import time


def profiler(fn):
    def timer(*args):
        t = time.time()
        result = fn(*args)
        print(f'[PROFILER] "{fn.__name__}": {time.time() - t:.2f}s')
        return result

    return timer


class CardImgGenerator(object):
    def __init__(self):
        self.img_size = (800, 600)
        self.card_width_range = (200, 400)
        self.amount = 10000

        self.backgrounds_dir = 'card_finder/src_images/backgrounds/'
        self.cards_dir = 'card_finder/src_images/cards/'

    def get_batch(self, amount=None):
        return self.generate(amount or self.amount)

    @profiler
    def generate(self, amount):
        imgs = np.zeros((amount, self.img_size[1], self.img_size[0], 3))
        borders = np.zeros((amount, 4))

        for i in range(amount):
            img, border = self.generate_img()
            imgs[i] = img
            borders[i] = border

        return imgs, borders

    @profiler
    def generate_img(self):
        bg = self.get_background_img()
        card = self.get_card_img()

        card_w = random.randint(*self.card_width_range)
        card_h = int((card_w / card.width) * card.height)

        t = time.time()
        card = card.resize((card_w, card_h))  # TODO: too slow!!!
        print(f'[PROFILER] "card.resize": {time.time() - t:.2f}s')

        x = 0 - random.randint(0, bg.width - card.width)
        y = 0 - random.randint(0, bg.height - card.height)

        card = card.transform(bg.size, Image.EXTENT, (x, y, bg.width + x, bg.height + y))

        result = Image.alpha_composite(bg, card)
        result = result.convert('RGB')
        borders = (-x, -y, card.width, card.height)

        return result, borders

    def get_background_img(self):
        fname = random.choice(os.listdir(self.backgrounds_dir))
        fpath = self.backgrounds_dir + fname

        img = Image.open(fpath)
        img = img.convert('RGBA')
        img = img.resize(self.img_size)

        return img

    def get_card_img(self):
        fname = random.choice(os.listdir(self.cards_dir))
        fpath = self.cards_dir + fname

        img = Image.open(fpath)

        return img
