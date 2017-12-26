import os
import random

import numpy as np
from PIL import Image


class CardImgGenerator(object):
    def __init__(self):
        self.img_size = (600, 450)
        self.card_width_range = (100, 300)
        self.amount = 10000

        self.backgrounds_dir = 'card_finder/src_images/backgrounds/'
        self.cards_dir = 'card_finder/src_images/cards/'

    def get_batch(self, amount=None):
        return self.generate(amount or self.amount)

    def generate(self, amount):
        imgs = np.zeros((amount, self.img_size[1], self.img_size[0], 3))
        borders = np.zeros((amount, 4))

        for i in range(amount):
            img, border = self.generate_img()
            imgs[i] = img
            borders[i] = border

        return imgs, borders

    def generate_img(self):
        bg = self.get_background_img()
        card = self.get_card_img()

        card_w = random.randint(*self.card_width_range)
        card_h = int((card_w / card.width) * card.height)

        card = card.resize((card_w, card_h))

        x = 0 - random.randint(0, bg.width - card.width)
        y = 0 - random.randint(0, bg.height - card.height)

        card = card.transform(bg.size, Image.EXTENT, (x, y, bg.width + x, bg.height + y))

        result = Image.alpha_composite(bg, card)
        result = result.convert('RGB')
        borders = (-x, -y, card_w, card_h)

        return result, borders

    def get_background_img(self):
        fname = random.choice(os.listdir(self.backgrounds_dir))
        fpath = self.backgrounds_dir + fname

        img = Image.open(fpath)
        img = img.resize(self.img_size)
        img = img.convert('RGBA')

        return img

    def get_card_img(self):
        fname = random.choice(os.listdir(self.cards_dir))
        fpath = self.cards_dir + fname

        img = Image.open(fpath)

        return img
